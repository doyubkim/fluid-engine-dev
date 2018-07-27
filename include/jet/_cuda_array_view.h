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
    CudaArrayView();

    CudaArrayView(T* ptr, const CudaStdArray<size_t, N>& size_);

    template <size_t M = N>
    CudaArrayView(typename std::enable_if<(M == 1), T>::type* ptr,
                  size_t size_);

    CudaArrayView(CudaArray<T, N>& other);

    CudaArrayView(const CudaArrayView& other);

    CudaArrayView(CudaArrayView&& other) noexcept;

    // set

    void set(CudaArray<T, N>& other);

    void set(const CudaArrayView& other);

    void fill(const T& val);

    // Assignment Operators
    CudaArrayView& operator=(const CudaArrayView& other);

    CudaArrayView& operator=(CudaArrayView&& other) noexcept;
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
    CudaArrayView();

    CudaArrayView(const T* ptr, const CudaStdArray<size_t, N>& size_);

    template <size_t M = N>
    CudaArrayView(const typename std::enable_if<(M == 1), T>::type* ptr,
                  size_t size_);

    CudaArrayView(const CudaArray<T, N>& other);

    CudaArrayView(const CudaArrayView<T, N>& other);

    CudaArrayView(const CudaArrayView& other);

    CudaArrayView(CudaArrayView&&) noexcept;

    // set

    void set(const CudaArray<T, N>& other);

    void set(const CudaArrayView<T, N>& other);

    void set(const CudaArrayView& other);

    // Assignment Operators
    CudaArrayView& operator=(const CudaArrayView<T, N>& other);

    CudaArrayView& operator=(const CudaArrayView& other);

    CudaArrayView& operator=(CudaArrayView&& other) noexcept;
};

template <class T>
using NewCudaArrayView1 = CudaArrayView<T, 1>;

template <class T>
using NewCudaArrayView2 = CudaArrayView<T, 2>;

template <class T>
using NewCudaArrayView3 = CudaArrayView<T, 3>;

template <class T>
using NewCudaArrayView4 = CudaArrayView<T, 4>;

template <class T>
using NewConstCudaArrayView1 = CudaArrayView<const T, 1>;

template <class T>
using NewConstCudaArrayView2 = CudaArrayView<const T, 2>;

template <class T>
using NewConstCudaArrayView3 = CudaArrayView<const T, 3>;

template <class T>
using NewConstCudaArrayView4 = CudaArrayView<const T, 4>;

}  // namespace jet

#include <jet/detail/_cuda_array_view-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_ARRAY_VIEW_H_
