// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_STD_ARRAY_H_
#define INCLUDE_JET_CUDA_STD_ARRAY_H_

#ifdef JET_USE_CUDA

#include <jet/matrix.h>

#include <array>

namespace jet {

template <typename T, size_t N>
class CudaStdArray {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    __host__ __device__ CudaStdArray();

    template <typename... Args>
    __host__ __device__ CudaStdArray(const_reference first, Args... rest);

    __host__ CudaStdArray(const std::array<T, N>& other);

    __host__ CudaStdArray(const Vector<T, N>& other);

    __host__ __device__ CudaStdArray(const CudaStdArray& other);

    __host__ __device__ void fill(const_reference val);

    __host__ __device__ reference operator[](size_t i);

    __host__ __device__ const_reference operator[](size_t i) const;

 private:
    T _elements[N];

    template <typename... Args>
    __host__ __device__ void setAt(size_t i, const_reference first,
                                   Args... rest);

    template <typename... Args>
    __host__ __device__ void setAt(size_t i, const_reference first);
};

}  // namespace jet

#include <jet/detail/cuda_std_array-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_STD_ARRAY_H_
