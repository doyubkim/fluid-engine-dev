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
                          ExecutionPolicy policy);

template <typename IndexType, typename Func>
void parallelForEachIndex(IndexType begin, IndexType end, Func func,
                          ExecutionPolicy policy);

template <typename IndexType, size_t N, typename Func>
void parallelForEachIndex(const Vector<IndexType, N>& size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename IndexType, typename Func>
void parallelForEachIndex(const Vector<IndexType, 1>& size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename IndexType, typename Func>
void parallelForEachIndex(IndexType size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

}  // namespace jet

#include <jet/detail/iteration_utils-inl.h>

#endif  // INCLUDE_JET_ITERATION_UTILS_H_
