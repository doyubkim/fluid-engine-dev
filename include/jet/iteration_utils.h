// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ITERATION_UTILS_H_
#define INCLUDE_JET_ITERATION_UTILS_H_

#include <jet/parallel.h>
#include <jet/tuple.h>

namespace jet {

// MARK: Serial Iteration

template <size_t N, typename Func>
void forEachIndex(const SizeN<N>& begin, const SizeN<N>& end, Func func);

template <typename Func>
void forEachIndex(const Size1& begin, const Size1& end, Func func);

template <typename Func>
void forEachIndex(size_t begin, size_t end, Func func);

template <size_t N, typename Func>
void forEachIndex(const SizeN<N>& size, Func func);

template <typename Func>
void forEachIndex(const Size1& size, Func func);

template <typename Func>
void forEachIndex(size_t size, Func func);

// MARK: Parallel Iteration

template <size_t N, typename Func>
void parallelForEachIndex(const SizeN<N>& begin, const SizeN<N>& end, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename Func>
void parallelForEachIndex(const Size1& begin, const Size1& end, Func func,
                          ExecutionPolicy policy);

template <typename Func>
void parallelForEachIndex(size_t begin, size_t end, Func func,
                          ExecutionPolicy policy);

template <size_t N, typename Func>
void parallelForEachIndex(const SizeN<N>& size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename Func>
void parallelForEachIndex(const Size1& size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

template <typename Func>
void parallelForEachIndex(size_t size, Func func,
                          ExecutionPolicy policy = ExecutionPolicy::kParallel);

}  // namespace jet

#include <jet/detail/iteration_utils-inl.h>

#endif  // INCLUDE_JET_ITERATION_UTILS_H_
