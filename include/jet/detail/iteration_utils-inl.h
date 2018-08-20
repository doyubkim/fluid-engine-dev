// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ITERATION_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_ITERATION_UTILS_INL_H_

#include <jet/iteration_utils.h>
#include <jet/parallel.h>

namespace jet {

namespace internal {

template <typename IndexType, size_t N, size_t I>
struct ForEachIndex {
    template <typename Func, typename... RemainingIndices>
    static void call(const Vector<IndexType, N>& begin,
                     const Vector<IndexType, N>& end, Func func,
                     RemainingIndices... indices) {
        for (IndexType i = begin[I - 1]; i < end[I - 1]; ++i) {
            ForEachIndex<IndexType, N, I - 1>::call(begin, end, func, i,
                                                    indices...);
        }
    }
};

template <typename IndexType, size_t N>
struct ForEachIndex<IndexType, N, 1> {
    template <typename Func, typename... RemainingIndices>
    static void call(const Vector<IndexType, N>& begin,
                     const Vector<IndexType, N>& end, Func func,
                     RemainingIndices... indices) {
        for (IndexType i = begin[0]; i < end[0]; ++i) {
            func(i, indices...);
        }
    }
};

}  // namespace internal

// MARK: Serial Iteration

template <typename IndexType, size_t N, typename Func>
void forEachIndex(const Vector<IndexType, N>& begin,
                  const Vector<IndexType, N>& end, Func func) {
    for (IndexType i = begin[N - 1]; i < end[N - 1]; ++i) {
        internal::ForEachIndex<IndexType, N, N - 1>::call(begin, end, func, i);
    }
}

template <typename IndexType, typename Func>
void forEachIndex(const Vector<IndexType, 1>& begin,
                  const Vector<IndexType, 1>& end, Func func) {
    for (IndexType i = begin[0]; i < end[0]; ++i) {
        func(i);
    }
}

template <typename IndexType, typename Func>
void forEachIndex(IndexType begin, IndexType end, Func func) {
    for (IndexType i = begin; i < end; ++i) {
        func(i);
    }
}

template <typename IndexType, size_t N, typename Func>
void forEachIndex(const Vector<IndexType, N>& size, Func func) {
    forEachIndex(Vector<IndexType, N>{}, size, func);
}

template <typename IndexType, typename Func>
void forEachIndex(const Vector<IndexType, 1>& size, Func func) {
    forEachIndex(Vector<IndexType, 1>{}, size, func);
}

template <typename IndexType, typename Func>
void forEachIndex(IndexType size, Func func) {
    forEachIndex(IndexType{}, size, func);
}

// MARK: Parallel Iteration

template <typename IndexType, size_t N, typename Func>
void parallelForEachIndex(const Vector<IndexType, N>& begin,
                          const Vector<IndexType, N>& end, Func func,
                          ExecutionPolicy policy) {
    parallelFor(begin[N - 1], end[N - 1],
                [&](IndexType i) {
                    internal::ForEachIndex<IndexType, N, N - 1>::call(
                        begin, end, func, i);
                },
                policy);
}

template <typename IndexType, typename Func>
void parallelForEachIndex(const Vector<IndexType, 1>& begin,
                          const Vector<IndexType, 1>& end, Func func,
                          ExecutionPolicy policy) {
    parallelFor(begin[0], end[0], func, policy);
}

template <typename IndexType, typename Func>
void parallelForEachIndex(IndexType begin, IndexType end, Func func,
                          ExecutionPolicy policy) {
    parallelFor(begin, end, func, policy);
}

template <typename IndexType, size_t N, typename Func>
void parallelForEachIndex(const Vector<IndexType, N>& size, Func func,
                          ExecutionPolicy policy) {
    parallelForEachIndex(Vector<IndexType, N>{}, size, func, policy);
}

template <typename IndexType, typename Func>
void parallelForEachIndex(const Vector<IndexType, 1>& size, Func func,
                          ExecutionPolicy policy) {
    parallelForEachIndex(Vector<IndexType, 1>{}, size, func, policy);
}

template <typename IndexType, typename Func>
void parallelForEachIndex(IndexType size, Func func, ExecutionPolicy policy) {
    parallelForEachIndex(IndexType{}, size, func, policy);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ITERATION_UTILS_INL_H_
