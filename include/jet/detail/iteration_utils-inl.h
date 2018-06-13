// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ITERATION_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_ITERATION_UTILS_INL_H_

#include <jet/parallel.h>

namespace jet {

namespace internal {

template <size_t N, size_t I>
struct ForEachIndex {
    template <typename Func, typename... RemainingIndices>
    static void call(const SizeN<N>& begin, const SizeN<N>& end, Func func,
                     RemainingIndices... indices) {
        for (size_t i = begin[I - 1]; i < end[I - 1]; ++i) {
            ForEachIndex<N, I - 1>::call(begin, end, func, i, indices...);
        }
    }
};

template <size_t N>
struct ForEachIndex<N, 1> {
    template <typename Func, typename... RemainingIndices>
    static void call(const SizeN<N>& begin, const SizeN<N>& end, Func func,
                     RemainingIndices... indices) {
        for (size_t i = begin[0]; i < end[0]; ++i) {
            func(i, indices...);
        }
    }
};

}  // namespace internal

// MARK: Serial Iteration

template <size_t N, typename Func>
void forEachIndex(const SizeN<N>& begin, const SizeN<N>& end, Func func) {
    for (size_t i = begin[N - 1]; i < end[N - 1]; ++i) {
        internal::ForEachIndex<N, N - 1>::call(begin, end, func, i);
    }
}

template <typename Func>
void forEachIndex(const Size1& begin, const Size1& end, Func func) {
    for (size_t i = begin[0]; i < end[0]; ++i) {
        func(i);
    }
}

template <typename Func>
void forEachIndex(size_t begin, size_t end, Func func) {
    for (size_t i = begin; i < end; ++i) {
        func(i);
    }
}

template <size_t N, typename Func>
void forEachIndex(SizeN<N> size, Func func) {
    forEachIndex({}, size, func);
}

template <typename Func>
void forEachIndex(Size1 size, Func func) {
    forEachIndex({}, size, func);
}

template <typename Func>
void forEachIndex(size_t size, Func func) {
    forEachIndex({}, {size}, func);
}

// MARK: Parallel Iteration

template <size_t N, typename Func>
void parallelForEachIndex(const SizeN<N>& begin, const SizeN<N>& end, Func func,
                          ExecutionPolicy policy) {
    parallelFor(begin[N - 1], end[N - 1],
                [&](size_t i) {
                    internal::ForEachIndex<N, N - 1>::call(begin, end, func, i);
                },
                policy);
}

template <typename Func>
void parallelForEachIndex(const Size1& begin, const Size1& end, Func func,
                          ExecutionPolicy policy) {
    parallelFor(begin[0], end[0], func, policy);
}

template <typename Func>
void parallelForEachIndex(size_t begin, size_t end, Func func,
                          ExecutionPolicy policy) {
    parallelFor(begin, end, func, policy);
}

template <size_t N, typename Func>
void parallelForEachIndex(SizeN<N> size, Func func, ExecutionPolicy policy) {
    parallelForEachIndex({}, size, func, policy);
}

template <typename Func>
void parallelForEachIndex(Size1 size, Func func, ExecutionPolicy policy) {
    parallelForEachIndex({}, size, func, policy);
}

template <typename Func>
void parallelForEachIndex(size_t size, Func func, ExecutionPolicy policy) {
    parallelForEachIndex({}, {size}, func, policy);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ITERATION_UTILS_INL_H_
