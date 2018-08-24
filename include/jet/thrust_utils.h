// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_THRUST_UTILS_H_
#define INCLUDE_JET_THRUST_UTILS_H_

#include <jet/cuda_array.h>

JET_DISABLE_CLANG_WARNING(unneeded-internal-declaration)
#include <thrust/device_ptr.h>
JET_ENABLE_CLANG_WARNING(unneeded-internal-declaration)

#include <cuda_runtime.h>

namespace jet {

template <typename T, size_t N, typename D>
thrust::device_ptr<T> thrustBegin(CudaArrayBase<T, N, D>& arr) {
    return thrust::device_ptr<T>(arr.data());
}

template <typename T, size_t N, typename D>
thrust::device_ptr<const T> thrustCBegin(const CudaArrayBase<T, N, D>& arr) {
    return thrust::device_ptr<const T>(arr.data());
}

template <typename T, size_t N, typename D>
thrust::device_ptr<T> thrustEnd(CudaArrayBase<T, N, D>& arr) {
    return thrust::device_ptr<T>(arr.data() + arr.length());
}

template <typename T, size_t N, typename D>
thrust::device_ptr<const T> thrustCEnd(const CudaArrayBase<T, N, D>& arr) {
    return thrust::device_ptr<const T>(arr.data() + arr.length());
}

}  // namespace jet

#endif  // INCLUDE_JET_THRUST_UTILS_H_

#endif  // JET_USE_CUDA
