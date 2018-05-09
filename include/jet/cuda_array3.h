// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY3_H_
#define INCLUDE_JET_CUDA_ARRAY3_H_

#include <jet/array_view3.h>
#include <jet/cuda_array1.h>
#include <jet/macros.h>

#include <thrust/device_vector.h>

#include <vector>

namespace jet {

template <typename T, size_t N>
class CudaArrayView;

template <typename T, size_t N>
class ConstCudaArrayView;

template <typename T>
class CudaArray<T, 3> final {
 public:
};

}  // namespace jet

#include "detail/cuda_array3-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY3_H_

#endif  // JET_USE_CUDA
