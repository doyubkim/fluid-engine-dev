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

    JET_CUDA_HOST_DEVICE CudaStdArray();

    template <typename... Args>
    JET_CUDA_HOST_DEVICE CudaStdArray(const_reference first, Args... rest);

    JET_CUDA_HOST CudaStdArray(const std::array<T, N>& other);

    JET_CUDA_HOST CudaStdArray(const Vector<T, N>& other);

    JET_CUDA_HOST_DEVICE CudaStdArray(const CudaStdArray& other);

    JET_CUDA_HOST_DEVICE void fill(const_reference val);

    JET_CUDA_HOST Vector<T, N> toVector() const;

    JET_CUDA_HOST_DEVICE reference operator[](size_t i);

    JET_CUDA_HOST_DEVICE const_reference operator[](size_t i) const;

    JET_CUDA_HOST_DEVICE bool operator==(const CudaStdArray& other) const;

    JET_CUDA_HOST_DEVICE bool operator!=(const CudaStdArray& other) const;

 private:
    T _elements[N];

    template <typename... Args>
    JET_CUDA_HOST_DEVICE void setAt(size_t i, const_reference first,
                                    Args... rest);

    template <typename... Args>
    JET_CUDA_HOST_DEVICE void setAt(size_t i, const_reference first);
};

}  // namespace jet

#include <jet/detail/cuda_std_array-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_STD_ARRAY_H_
