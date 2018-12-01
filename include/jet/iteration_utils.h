// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ITERATION_UTILS_H_
#define INCLUDE_JET_ITERATION_UTILS_H_

#include <jet/matrix.h>
#include <jet/parallel.h>

namespace jet {

// MARK: Serial Iteration

template <typename IndexType, size_t N, typename Func>
void forEachIndex(const Vector<IndexType, N>& begin,
                  const Vector<IndexType, N>& end, Func func);

template <typename IndexType, typename Func>
void forEachIndex(const Vector<IndexType, 1>& begin,
                  const Vector<IndexType, 1>& end, Func func);

template <typename IndexType, typename Func>
void forEachIndex(IndexType begin, IndexType end, Func func);

template <typename IndexType, size_t N, typename Func>
void forEachIndex(const Vector<IndexType, N>& size, Func func);

template <typename IndexType, typename Func>
void forEachIndex(const Vector<IndexType, 1>& size, Func func);

template <typename IndexType, typename Func>
void forEachIndex(IndexType size, Func func);

// MARK: Parallel Iteration

template <typename IndexType, size_t N, typename Func>
void parallelForEachIndex(const Vector<IndexType, N>& begin,
                          const Vector<IndexType, N>& end, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename IndexType, typename Func>
void parallelForEachIndex(const Vector<IndexType, 1>& begin,
                          const Vector<IndexType, 1>& end, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename IndexType, typename Func>
void parallelForEachIndex(IndexType begin, IndexType end, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename IndexType, size_t N, typename Func>
void parallelForEachIndex(const Vector<IndexType, N>& size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename IndexType, typename Func>
void parallelForEachIndex(const Vector<IndexType, 1>& size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename IndexType, typename Func>
void parallelForEachIndex(IndexType size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

// MARK: Iteration Adapters

//! Unrolls vector-based indexing to size_t-based function.
template <typename ReturnType>
inline std::function<ReturnType(size_t)> unroll1(
    const std::function<ReturnType(const Vector1UZ&)>& func) {
    return [func](size_t i) { return func(Vector1UZ(i)); };
}

//! Unrolls vector-based indexing to size_t-based function.
template <typename ReturnType>
inline std::function<ReturnType(size_t, size_t)> unroll2(
    const std::function<ReturnType(const Vector2UZ&)>& func) {
    return [func](size_t i, size_t j) { return func(Vector2UZ(i, j)); };
}

//! Unrolls vector-based DataPositionFunc indexing to size_t-based function.
template <typename ReturnType>
inline std::function<ReturnType(size_t, size_t, size_t)> unroll3(
    const std::function<ReturnType(const Vector3UZ&)>& func) {
    return [func](size_t i, size_t j, size_t k) {
        return func(Vector3UZ(i, j, k));
    };
}

template <typename ReturnType, size_t N>
struct GetUnroll {};

template <typename ReturnType>
struct GetUnroll<ReturnType, 1> {
    static std::function<ReturnType(size_t)> unroll(
        const std::function<ReturnType(const Vector1UZ&)>& func) {
        return [func](size_t i) { return func(Vector1UZ(i)); };
    }
};

template <>
struct GetUnroll<void, 1> {
    static std::function<void(size_t)> unroll(
        const std::function<void(const Vector1UZ&)>& func) {
        return [func](size_t i) { func(Vector1UZ(i)); };
    }
};

template <typename ReturnType>
struct GetUnroll<ReturnType, 2> {
    static std::function<ReturnType(size_t, size_t)> unroll(
        const std::function<ReturnType(const Vector2UZ&)>& func) {
        return [func](size_t i, size_t j) { return func(Vector2UZ(i, j)); };
    }
};

template <>
struct GetUnroll<void, 2> {
    static std::function<void(size_t, size_t)> unroll(
        const std::function<void(const Vector2UZ&)>& func) {
        return [func](size_t i, size_t j) { func(Vector2UZ(i, j)); };
    }
};

template <typename ReturnType>
struct GetUnroll<ReturnType, 3> {
    static std::function<ReturnType(size_t, size_t, size_t)> unroll(
        const std::function<ReturnType(const Vector3UZ&)>& func) {
        return [func](size_t i, size_t j, size_t k) {
            return func(Vector3UZ(i, j, k));
        };
    }
};

template <>
struct GetUnroll<void, 3> {
    static std::function<void(size_t, size_t, size_t)> unroll(
        const std::function<void(const Vector3UZ&)>& func) {
        return [func](size_t i, size_t j, size_t k) {
            return func(Vector3UZ(i, j, k));
        };
    }
};

}  // namespace jet

#include <jet/detail/iteration_utils-inl.h>

#endif  // INCLUDE_JET_ITERATION_UTILS_H_
