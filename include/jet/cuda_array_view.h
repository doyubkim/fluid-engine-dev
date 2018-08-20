// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_ARRAY_VIEW_H_
#define INCLUDE_JET_CUDA_ARRAY_VIEW_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_array.h>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaArrayView

template <typename T, size_t N>
class CudaArrayView final : public CudaArrayBase<T, N, CudaArrayView<T, N>> {
    using Base = CudaArrayBase<T, N, CudaArrayView<T, N>>;
    using Base::_size;
    using Base::at;
    using Base::setPtrAndSize;

 public:
    using Base::data;

    // CTOR
    __host__ __device__ CudaArrayView();

    __host__ __device__ CudaArrayView(T* ptr,
                                      const CudaStdArray<size_t, N>& size_);

    template <size_t M = N>
    __host__ __device__ CudaArrayView(
        typename std::enable_if<(M == 1), T>::type* ptr, size_t size_);

    __host__ __device__ CudaArrayView(CudaArray<T, N>& other);

    __host__ __device__ CudaArrayView(const CudaArrayView& other);

    __host__ __device__ CudaArrayView(CudaArrayView&& other) noexcept;

    // set

    __host__ __device__ void set(CudaArray<T, N>& other);

    __host__ __device__ void set(const CudaArrayView& other);

    __host__ void fill(const T& val);

    // Assignment Operators
    __host__ __device__ CudaArrayView& operator=(const CudaArrayView& other);

    __host__ __device__ CudaArrayView& operator=(
        CudaArrayView&& other) noexcept;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: Immutable CudaArrayView Specialization for CUDA

template <typename T, size_t N>
class CudaArrayView<const T, N> final
    : public CudaArrayBase<const T, N, CudaArrayView<const T, N>> {
    using Base = CudaArrayBase<const T, N, CudaArrayView<const T, N>>;
    using Base::_size;
    using Base::setPtrAndSize;

 public:
    using Base::data;

    // CTOR
    __host__ __device__ CudaArrayView();

    __host__ __device__ CudaArrayView(const T* ptr,
                                      const CudaStdArray<size_t, N>& size_);

    template <size_t M = N>
    __host__ __device__ CudaArrayView(
        const typename std::enable_if<(M == 1), T>::type* ptr, size_t size_);

    __host__ __device__ CudaArrayView(const CudaArray<T, N>& other);

    __host__ __device__ CudaArrayView(const CudaArrayView<T, N>& other);

    __host__ __device__ CudaArrayView(const CudaArrayView& other);

    __host__ __device__ CudaArrayView(CudaArrayView&&) noexcept;

    // set

    __host__ __device__ void set(const CudaArray<T, N>& other);

    __host__ __device__ void set(const CudaArrayView<T, N>& other);

    __host__ __device__ void set(const CudaArrayView& other);

    // Assignment Operators
    __host__ __device__ CudaArrayView& operator=(
        const CudaArrayView<T, N>& other);

    __host__ __device__ CudaArrayView& operator=(const CudaArrayView& other);

    __host__ __device__ CudaArrayView& operator=(
        CudaArrayView&& other) noexcept;
};

template <class T>
using CudaArrayView1 = CudaArrayView<T, 1>;

template <class T>
using CudaArrayView2 = CudaArrayView<T, 2>;

template <class T>
using CudaArrayView3 = CudaArrayView<T, 3>;

template <class T>
using CudaArrayView4 = CudaArrayView<T, 4>;

template <class T>
using ConstCudaArrayView1 = CudaArrayView<const T, 1>;

template <class T>
using ConstCudaArrayView2 = CudaArrayView<const T, 2>;

template <class T>
using ConstCudaArrayView3 = CudaArrayView<const T, 3>;

template <class T>
using ConstCudaArrayView4 = CudaArrayView<const T, 4>;

}  // namespace jet

#include <jet/detail/cuda_array_view-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_ARRAY_VIEW_H_
