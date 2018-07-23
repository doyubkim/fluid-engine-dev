// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_ARRAY_VIEW_H_
#define INCLUDE_JET_CUDA_ARRAY_VIEW_H_

#ifdef JET_USE_CUDA

#include <jet/_cuda_array.h>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: ArrayView Specialization for CUDA

template <typename T, size_t N>
class ArrayView<T, N, CudaDevice<T>> final
    : public ArrayBase<T, N, CudaDevice<T>, ArrayView<T, N, CudaDevice<T>>> {
    using Device = CudaDevice<T>;
    using Base = ArrayBase<T, N, Device, ArrayView<T, N, Device>>;
    using Base::_size;
    using Base::at;
    using Base::setHandleAndSize;

 public:
    // CTOR
    __host__ __device__ ArrayView();

    __host__ __device__ ArrayView(T* ptr,
                                  const thrust::array<size_t, N>& size_);

    template <size_t M = N>
    __host__ __device__
    ArrayView(typename std::enable_if<(M == 1), T>::type* ptr, size_t size_);

    __host__ __device__ ArrayView(Array<T, N, Device>& other);

    __host__ __device__ ArrayView(const ArrayView& other);

    __host__ __device__ ArrayView(ArrayView&& other) noexcept;

    // set

    __host__ __device__ void set(Array<T, N, Device>& other);

    __host__ __device__ void set(const ArrayView& other);

    __host__ void fill(const T& val);

    // Assignment Operators
    __host__ __device__ ArrayView& operator=(const ArrayView& other);

    __host__ __device__ ArrayView& operator=(ArrayView&& other) noexcept;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: Immutable ArrayView Specialization for CUDA

template <typename T, size_t N>
class ArrayView<const T, N, CudaDevice<T>> final
    : public ArrayBase<const T, N, CudaDevice<T>,
                       ArrayView<const T, N, CudaDevice<T>>> {
    using Device = CudaDevice<T>;
    using Base = ArrayBase<const T, N, Device, ArrayView<const T, N, Device>>;
    using Base::_size;
    using Base::setHandleAndSize;

 public:
    // CTOR
    __host__ __device__ ArrayView();

    __host__ __device__ ArrayView(const T* ptr,
                                  const thrust::array<size_t, N>& size_);

    template <size_t M = N>
    __host__ __device__ ArrayView(
        const typename std::enable_if<(M == 1), T>::type* ptr, size_t size_);

    __host__ __device__ ArrayView(const Array<T, N, Device>& other);

    __host__ __device__ ArrayView(const ArrayView<T, N, Device>& other);

    __host__ __device__ ArrayView(const ArrayView& other);

    __host__ __device__ ArrayView(ArrayView&&) noexcept;

    // set

    __host__ __device__ void set(const Array<T, N, Device>& other);

    __host__ __device__ void set(const ArrayView<T, N, Device>& other);

    __host__ __device__ void set(const ArrayView& other);

    // Assignment Operators
    __host__ __device__ ArrayView& operator=(
        const ArrayView<T, N, Device>& other);

    __host__ __device__ ArrayView& operator=(const ArrayView& other);

    __host__ __device__ ArrayView& operator=(ArrayView&& other) noexcept;
};

template <class T>
using NewCudaArrayView1 = ArrayView<T, 1, CudaDevice<T>>;

template <class T>
using NewCudaArrayView2 = ArrayView<T, 2, CudaDevice<T>>;

template <class T>
using NewCudaArrayView3 = ArrayView<T, 3, CudaDevice<T>>;

template <class T>
using NewCudaArrayView4 = ArrayView<T, 4, CudaDevice<T>>;

template <class T>
using NewConstCudaArrayView1 = ArrayView<const T, 1, CudaDevice<T>>;

template <class T>
using NewConstCudaArrayView2 = ArrayView<const T, 2, CudaDevice<T>>;

template <class T>
using NewConstCudaArrayView3 = ArrayView<const T, 3, CudaDevice<T>>;

template <class T>
using NewConstCudaArrayView4 = ArrayView<const T, 4, CudaDevice<T>>;

}  // namespace jet

#include <jet/detail/_cuda_array_view-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_ARRAY_VIEW_H_
