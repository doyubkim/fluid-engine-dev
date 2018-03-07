// Copyright (c) 2018 Doyub Kim
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
#include <future>
#include <vector>

#ifdef JET_TASKING_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/task.h>
#elif defined(JET_TASKING_CPP11THREADS)
#include <thread>
#endif

namespace jet {

namespace internal {

// NOTE - This abstraction takes a lambda which should take captured
//        variables by *value* to ensure no captured references race
//        with the task itself.
template <typename TASK_T>
inline void schedule(TASK_T&& fcn) {
#ifdef JET_TASKING_TBB
    struct LocalTBBTask : public tbb::task {
        TASK_T func;
        tbb::task* execute() override {
            func();
            return nullptr;
        }
        LocalTBBTask(TASK_T&& f) : func(std::forward<TASK_T>(f)) {}
    };

    auto* tbb_node = new (tbb::task::allocate_root())
        LocalTBBTask(std::forward<TASK_T>(fcn));
    tbb::task::enqueue(*tbb_node);
#elif defined(JET_TASKING_CPP11THREADS)
    std::thread thread(fcn);
    thread.detach();
#else  // OpenMP or Serial --> synchronous!
    fcn();
#endif
}

template <typename TASK_T>
using operator_return_t = typename std::result_of<TASK_T()>::type;

// NOTE - see above, same issues associated with schedule()
template <typename TASK_T>
inline auto async(TASK_T&& fcn) -> std::future<operator_return_t<TASK_T>> {
    using package_t = std::packaged_task<operator_return_t<TASK_T>()>;

    auto task = new package_t(std::forward<TASK_T>(fcn));
    auto future = task->get_future();

    schedule([=]() {
        (*task)();
        delete task;
    });

    return future;
}

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
        std::vector<std::future<void>> pool;
        pool.reserve(2);

        auto launchRange = [compareFunction](RandomIterator begin, size_t k2,
                                             RandomIterator2 temp,
                                             unsigned int numThreads) {
            parallelMergeSort(begin, k2, temp, numThreads, compareFunction);
        };

        pool.emplace_back(internal::async(
            [=]() { launchRange(a, size / 2, temp, numThreads / 2); }));

        pool.emplace_back(internal::async([=]() {
            launchRange(a + size / 2, size - size / 2, temp + size / 2,
                        numThreads - numThreads / 2);
        }));

        // Wait for jobs to finish
        for (auto& f : pool) {
            if (f.valid()) {
                f.wait();
            }
        }

        merge(a, size, temp, compareFunction);
    }
}

}  // namespace internal

template <typename RandomIterator, typename T>
void parallelFill(const RandomIterator& begin, const RandomIterator& end,
                  const T& value, ExecutionPolicy policy) {
    auto diff = end - begin;
    if (diff <= 0) {
        return;
    }

    size_t size = static_cast<size_t>(diff);
    parallelFor(kZeroSize, size, [begin, value](size_t i) { begin[i] = value; },
                policy);
}

// Adopted from http://ideone.com/Z7zldb
template <typename IndexType, typename Function>
void parallelFor(IndexType start, IndexType end, const Function& func,
                 ExecutionPolicy policy) {
    if (start > end) {
        return;
    }

#ifdef JET_TASKING_TBB
    if (policy == ExecutionPolicy::kParallel) {
        tbb::parallel_for(start, end, func);
    } else {
        for (auto i = start; i < end; ++i) {
            func(i);
        }
    }

#elif JET_TASKING_CPP11THREADS
    // Estimate number of threads in the pool
    unsigned int numThreadsHint = maxNumberOfThreads();
    const unsigned int numThreads =
        (policy == ExecutionPolicy::kParallel)
            ? (numThreadsHint == 0u ? 8u : numThreadsHint)
            : 1;

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
#else

#ifdef JET_TASKING_OPENMP
    if (policy == ExecutionPolicy::kParallel) {
#pragma omp parallel for
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
        for (ssize_t i = start; i < ssize_t(end); ++i) {
#else   // !MSVC || Intel
        for (auto i = start; i < end; ++i) {
#endif  // MSVC && !Intel
            func(i);
        }
    } else {
        for (auto i = start; i < end; ++i) {
            func(i);
        }
    }
#else   // JET_TASKING_OPENMP
    for (auto i = start; i < end; ++i) {
        func(i);
    }
#endif  // JET_TASKING_OPENMP

#endif
}

template <typename IndexType, typename Function>
void parallelRangeFor(IndexType start, IndexType end, const Function& func,
                      ExecutionPolicy policy) {
    if (start > end) {
        return;
    }

#ifdef JET_TASKING_TBB
    if (policy == ExecutionPolicy::kParallel) {
        tbb::parallel_for(tbb::blocked_range<IndexType>(start, end),
                          [&func](const tbb::blocked_range<IndexType>& range) {
                              func(range.begin(), range.end());
                          });
    } else {
        func(start, end);
    }

#else
    // Estimate number of threads in the pool
    unsigned int numThreadsHint = maxNumberOfThreads();
    const unsigned int numThreads =
        (policy == ExecutionPolicy::kParallel)
            ? (numThreadsHint == 0u ? 8u : numThreadsHint)
            : 1;

    // Size of a slice for the range functions
    IndexType n = end - start + 1;
    IndexType slice =
        (IndexType)std::round(n / static_cast<double>(numThreads));
    slice = std::max(slice, IndexType(1));

    // Create pool and launch jobs
    std::vector<std::future<void>> pool;
    pool.reserve(numThreads);
    IndexType i1 = start;
    IndexType i2 = std::min(start + slice, end);
    for (unsigned int i = 0; i + 1 < numThreads && i1 < end; ++i) {
        pool.emplace_back(internal::async([=]() { func(i1, i2); }));
        i1 = i2;
        i2 = std::min(i2 + slice, end);
    }
    if (i1 < end) {
        pool.emplace_back(internal::async([=]() { func(i1, end); }));
    }

    // Wait for jobs to finish
    for (auto& f : pool) {
        if (f.valid()) {
            f.wait();
        }
    }
#endif
}

template <typename IndexType, typename Function>
void parallelFor(IndexType beginIndexX, IndexType endIndexX,
                 IndexType beginIndexY, IndexType endIndexY,
                 const Function& function, ExecutionPolicy policy) {
    parallelFor(beginIndexY, endIndexY,
                [&](IndexType j) {
                    for (IndexType i = beginIndexX; i < endIndexX; ++i) {
                        function(i, j);
                    }
                },
                policy);
}

template <typename IndexType, typename Function>
void parallelRangeFor(IndexType beginIndexX, IndexType endIndexX,
                      IndexType beginIndexY, IndexType endIndexY,
                      const Function& function, ExecutionPolicy policy) {
    parallelRangeFor(beginIndexY, endIndexY,
                     [&](IndexType jBegin, IndexType jEnd) {
                         function(beginIndexX, endIndexX, jBegin, jEnd);
                     },
                     policy);
}

template <typename IndexType, typename Function>
void parallelFor(IndexType beginIndexX, IndexType endIndexX,
                 IndexType beginIndexY, IndexType endIndexY,
                 IndexType beginIndexZ, IndexType endIndexZ,
                 const Function& function, ExecutionPolicy policy) {
    parallelFor(beginIndexZ, endIndexZ,
                [&](IndexType k) {
                    for (IndexType j = beginIndexY; j < endIndexY; ++j) {
                        for (IndexType i = beginIndexX; i < endIndexX; ++i) {
                            function(i, j, k);
                        }
                    }
                },
                policy);
}

template <typename IndexType, typename Function>
void parallelRangeFor(IndexType beginIndexX, IndexType endIndexX,
                      IndexType beginIndexY, IndexType endIndexY,
                      IndexType beginIndexZ, IndexType endIndexZ,
                      const Function& function, ExecutionPolicy policy) {
    parallelRangeFor(beginIndexZ, endIndexZ,
                     [&](IndexType kBegin, IndexType kEnd) {
                         function(beginIndexX, endIndexX, beginIndexY,
                                  endIndexY, kBegin, kEnd);
                     },
                     policy);
}

template <typename IndexType, typename Value, typename Function,
          typename Reduce>
Value parallelReduce(IndexType start, IndexType end, const Value& identity,
                     const Function& func, const Reduce& reduce,
                     ExecutionPolicy policy) {
    if (start > end) {
        return identity;
    }

#ifdef JET_TASKING_TBB
    if (policy == ExecutionPolicy::kParallel) {
        return tbb::parallel_reduce(
            tbb::blocked_range<IndexType>(start, end), identity,
            [&func](const tbb::blocked_range<IndexType>& range,
                    const Value& init) {
                return func(range.begin(), range.end(), init);
            },
            reduce);
    } else {
        (void)reduce;
        return func(start, end, identity);
    }

#else
    // Estimate number of threads in the pool
    unsigned int numThreadsHint = maxNumberOfThreads();
    const unsigned int numThreads =
        (policy == ExecutionPolicy::kParallel)
            ? (numThreadsHint == 0u ? 8u : numThreadsHint)
            : 1;

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
    std::vector<std::future<void>> pool;
    pool.reserve(numThreads);
    IndexType i1 = start;
    IndexType i2 = std::min(start + slice, end);
    unsigned int tid = 0;
    for (; tid + 1 < numThreads && i1 < end; ++tid) {
        pool.emplace_back(internal::async([=]() { launchRange(i1, i2, tid); }));
        i1 = i2;
        i2 = std::min(i2 + slice, end);
    }
    if (i1 < end) {
        pool.emplace_back(
            internal::async([=]() { launchRange(i1, end, tid); }));
    }

    // Wait for jobs to finish
    for (auto& f : pool) {
        if (f.valid()) {
            f.wait();
        }
    }

    // Gather
    Value finalResult = identity;
    for (const Value& val : results) {
        finalResult = reduce(val, finalResult);
    }

    return finalResult;
#endif
}

template <typename RandomIterator, typename CompareFunction>
void parallelSort(RandomIterator begin, RandomIterator end,
                  CompareFunction compareFunction, ExecutionPolicy policy) {
    if (end < begin) {
        return;
    }

#ifdef JET_TASKING_TBB
    if (policy == ExecutionPolicy::kParallel) {
        tbb::parallel_sort(begin, end, compareFunction);
    } else {
        std::sort(begin, end, compareFunction);
    }

#else
    size_t size = static_cast<size_t>(end - begin);

    typedef
        typename std::iterator_traits<RandomIterator>::value_type value_type;
    std::vector<value_type> temp(size);

    // Estimate number of threads in the pool
    unsigned int numThreadsHint = maxNumberOfThreads();
    const unsigned int numThreads =
        (policy == ExecutionPolicy::kParallel)
            ? (numThreadsHint == 0u ? 8u : numThreadsHint)
            : 1;

    internal::parallelMergeSort(begin, size, temp.begin(), numThreads,
                                compareFunction);
#endif
}

template <typename RandomIterator>
void parallelSort(RandomIterator begin, RandomIterator end,
                  ExecutionPolicy policy) {
    parallelSort(
        begin, end,
        std::less<typename std::iterator_traits<RandomIterator>::value_type>(),
        policy);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_PARALLEL_INL_H_
