// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_ALGORITHMS_H_
#define INCLUDE_JET_CUDA_ALGORITHMS_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_utils.h>

namespace jet {

#ifdef __CUDACC__

template <typename T>
__global__ void cudaFillKernel(T* dst, size_t n, T val) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = val;
    }
}

template <typename T>
void cudaFill(T* dst, size_t n, const T& val) {
    unsigned int numBlocks, numThreads;
    cudaComputeGridSize((unsigned int)n, 256, numBlocks, numThreads);
    cudaFillKernel<<<numBlocks, numThreads>>>(dst, n, val);
    JET_CUDA_CHECK_LAST_ERROR("Failed executing cudaFillKernel");
}

#endif  // __CUDACC__

template <typename T>
void cudaCopy(const T* src, size_t n, T* dst,
              cudaMemcpyKind kind = cudaMemcpyDeviceToDevice) {
    JET_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), kind));
}

template <typename T>
void cudaCopyDeviceToDevice(const T* src, size_t n, T* dst) {
    cudaCopy(src, n, dst, cudaMemcpyDeviceToDevice);
}

template <typename T>
void cudaCopyHostToDevice(const T* src, size_t n, T* dst) {
    cudaCopy(src, n, dst, cudaMemcpyHostToDevice);
}

template <typename T>
void cudaCopyDeviceToHost(const T* src, size_t n, T* dst) {
    cudaCopy(src, n, dst, cudaMemcpyDeviceToHost);
}

}  // namespace jet

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_ALGORITHMS_H_
