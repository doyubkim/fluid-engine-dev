// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_PARALLEL_INL_H_
#define INCLUDE_JET_DETAIL_PARALLEL_INL_H_

#include <jet/constants.h>
#include <jet/macros.h>

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

namespace jet {

namespace internal {

// Adopted from:
// Radenski, A.
// Shared Memory, Message Passing, and Hybrid Merge Sorts for Standalone and
// Clustered SMPs. Proc PDPTA'11, the  2011 International Conference on Parallel
// and Distributed Processing Techniques and Applications, CSREA Press
// (H. Arabnia, Ed.), 2011, pp. 367 - 373.
template <typename RandomIterator, typename RandomIterator2,
          typename CompareFunction>
void merge(RandomIterator a, size_t size, RandomIterator2 temp,
           CompareFunction compareFunction) {
    size_t i1 = 0;
    size_t i2 = size / 2;
    size_t tempi = 0;

    while (i1 < size / 2 && i2 < size) {
        if (compareFunction(a[i1], a[i2])) {
            temp[tempi] = a[i1];
            i1++;
        } else {
            temp[tempi] = a[i2];
            i2++;
        }
        tempi++;
    }

    while (i1 < size / 2) {
        temp[tempi] = a[i1];
        i1++;
        tempi++;
    }

    while (i2 < size) {
        temp[tempi] = a[i2];
        i2++;
        tempi++;
    }

    // Copy sorted temp array into main array, a
    parallelFor(kZeroSize, size, [&](size_t i) { a[i] = temp[i]; });
}

template <typename RandomIterator, typename RandomIterator2,
          typename CompareFunction>
void parallelMergeSort(RandomIterator a, size_t size, RandomIterator2 temp,
                       unsigned int numThreads,
                       CompareFunction compareFunction) {
    if (numThreads == 1) {
        std::sort(a, a + size, compareFunction);
    } else if (numThreads > 1) {
        std::vector<std::thread> pool;
        pool.reserve(2);

        auto launchRange = [compareFunction](RandomIterator begin, size_t k2,
                                             RandomIterator2 temp,
                                             unsigned int numThreads) {
            parallelMergeSort(begin, k2, temp, numThreads, compareFunction);
        };

        pool.emplace_back(launchRange, a, size / 2, temp, numThreads / 2);
        pool.emplace_back(launchRange, a + size / 2, size - size / 2,
                          temp + size / 2, numThreads - numThreads / 2);

        // Wait for jobs to finish
        for (std::thread& t : pool) {
            if (t.joinable()) {
                t.join();
            }
        }

        merge(a, size, temp, compareFunction);
    }
}

}  // namespace internal

template <typename RandomIterator, typename T>
void parallelFill(const RandomIterator& begin, const RandomIterator& end,
                  const T& value) {
    auto diff = end - begin;
    if (diff <= 0) {
        return;
    }

    size_t size = static_cast<size_t>(diff);
    parallelFor(kZeroSize, size,
                [begin, value](size_t i) { begin[i] = value; });
}

// Adopted from http://ideone.com/Z7zldb
template <typename IndexType, typename Function>
void parallelFor(IndexType start, IndexType end, const Function& func) {
    if (start > end) {
        return;
    }

    // Estimate number of threads in the pool
    static const unsigned int numThreadsHint =
        std::thread::hardware_concurrency();
    static const unsigned int numThreads =
        (numThreadsHint == 0u ? 8u : numThreadsHint);

    // Size of a slice for the range functions
    IndexType n = end - start + 1;
    IndexType slice =
        (IndexType)std::round(n / static_cast<double>(numThreads));
    slice = std::max(slice, IndexType(1));

    // [Helper] Inner loop
    auto launchRange = [&func](IndexType k1, IndexType k2) {
        for (IndexType k = k1; k < k2; k++) {
            func(k);
        }
    };

    // Create pool and launch jobs
    std::vector<std::thread> pool;
    pool.reserve(numThreads);
    IndexType i1 = start;
    IndexType i2 = std::min(start + slice, end);
    for (unsigned int i = 0; i + 1 < numThreads && i1 < end; ++i) {
        pool.emplace_back(launchRange, i1, i2);
        i1 = i2;
        i2 = std::min(i2 + slice, end);
    }
    if (i1 < end) {
        pool.emplace_back(launchRange, i1, end);
    }

    // Wait for jobs to finish
    for (std::thread& t : pool) {
        if (t.joinable()) {
            t.join();
        }
    }
}

template <typename IndexType, typename Function>
void parallelFor(IndexType beginIndexX, IndexType endIndexX,
                 IndexType beginIndexY, IndexType endIndexY,
                 const Function& function) {
    parallelFor(beginIndexY, endIndexY, [&](size_t j) {
        for (IndexType i = beginIndexX; i < endIndexX; ++i) {
            function(i, j);
        }
    });
}

template <typename IndexType, typename Function>
void parallelFor(IndexType beginIndexX, IndexType endIndexX,
                 IndexType beginIndexY, IndexType endIndexY,
                 IndexType beginIndexZ, IndexType endIndexZ,
                 const Function& function) {
    parallelFor(beginIndexZ, endIndexZ, [&](size_t k) {
        for (IndexType j = beginIndexY; j < endIndexY; ++j) {
            for (IndexType i = beginIndexX; i < endIndexX; ++i) {
                function(i, j, k);
            }
        }
    });
}

template <typename IndexType, typename Value, typename Function,
          typename Reduce>
Value parallelReduce(IndexType start, IndexType end, const Value& identity,
                     const Function& func, const Reduce& reduce) {
    if (start > end) {
        return identity;
    }

    // Estimate number of threads in the pool
    static const unsigned int numThreadsHint =
        std::thread::hardware_concurrency();
    static const unsigned int numThreads =
        (numThreadsHint == 0u ? 8u : numThreadsHint);

    // Size of a slice for the range functions
    IndexType n = end - start + 1;
    IndexType slice =
        (IndexType)std::round(n / static_cast<double>(numThreads));
    slice = std::max(slice, IndexType(1));

    // Results
    std::vector<Value> results(numThreads, identity);

    // [Helper] Inner loop
    auto launchRange = [&](IndexType k1, IndexType k2, unsigned int tid) {
        results[tid] = func(k1, k2, identity);
    };

    // Create pool and launch jobs
    std::vector<std::thread> pool;
    pool.reserve(numThreads);
    IndexType i1 = start;
    IndexType i2 = std::min(start + slice, end);
    unsigned int tid = 0;
    for (; tid + 1 < numThreads && i1 < end; ++tid) {
        pool.emplace_back(launchRange, i1, i2, tid);
        i1 = i2;
        i2 = std::min(i2 + slice, end);
    }
    if (i1 < end) {
        pool.emplace_back(launchRange, i1, end, tid);
    }

    // Wait for jobs to finish
    for (std::thread& t : pool) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Gather
    Value finalResult = identity;
    for (const Value& val : results) {
        finalResult = reduce(val, finalResult);
    }

    return finalResult;
}

template <typename RandomIterator, typename CompareFunction>
void parallelSort(RandomIterator begin, RandomIterator end,
                  CompareFunction compareFunction) {
    if (end < begin) {
        return;
    }

    size_t size = static_cast<size_t>(end - begin);

    typedef
        typename std::iterator_traits<RandomIterator>::value_type value_type;
    std::vector<value_type> temp(size);

    // Estimate number of threads in the pool
    static const unsigned int numThreadsHint =
        std::thread::hardware_concurrency();
    static const unsigned int numThreads =
        (numThreadsHint == 0u ? 8u : numThreadsHint);

    internal::parallelMergeSort(begin, size, temp.begin(), numThreads,
                                compareFunction);
}

template <typename RandomIterator>
void parallelSort(RandomIterator begin, RandomIterator end) {
    parallelSort(
        begin, end,
        std::less<typename std::iterator_traits<RandomIterator>::value_type>());
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_PARALLEL_INL_H_
