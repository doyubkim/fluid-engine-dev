// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_VIEW_H_
#define INCLUDE_JET_ARRAY_VIEW_H_

#include <jet/matrix.h>

namespace jet {

template <typename T>
struct CpuMemory;

#ifdef JET_USE_CUDA
template <typename T>
struct CudaMemory;
#endif

template <typename T, size_t N, typename Handle, typename Derived>
class ArrayBase;

template <typename T, size_t N, typename Handle>
class Array;

// MARK: ArrayView

template <typename T, size_t N, typename Handle>
class ArrayView final
    : public ArrayBase<T, N, Handle, ArrayView<T, N, Handle>> {
    using Base = ArrayBase<T, N, Handle, ArrayView<T, N, Handle>>;
    using Base::_size;
    using Base::at;
    using Base::setHandleAndSize;

 public:
    // CTOR
    ArrayView();

    ArrayView(T* ptr, const Vector<size_t, N>& size_);

    template <size_t M = N>
    ArrayView(typename std::enable_if<(M == 1), T>::type* ptr, size_t size_);

    ArrayView(Array<T, N, Handle>& other);

    ArrayView(const ArrayView& other);

    ArrayView(ArrayView&& other) noexcept;

    // set

    void set(Array<T, N, Handle>& other);

    void set(const ArrayView& other);

    void fill(const T& val);

    // Assignment Operators
    ArrayView& operator=(const ArrayView& other);

    ArrayView& operator=(ArrayView&& other) noexcept;
};

template <typename T, size_t N, typename Handle>
class ArrayView<const T, N, Handle> final
    : public ArrayBase<const T, N, Handle, ArrayView<const T, N, Handle>> {
    using Base = ArrayBase<const T, N, Handle, ArrayView<const T, N, Handle>>;
    using Base::_size;
    using Base::setHandleAndSize;

 public:
    // CTOR
    ArrayView();

    ArrayView(const T* ptr, const Vector<size_t, N>& size_);

    template <size_t M = N>
    ArrayView(const typename std::enable_if<(M == 1), T>::type* ptr,
              size_t size_);

    ArrayView(const Array<T, N, Handle>& other);

    ArrayView(const ArrayView<T, N, Handle>& other);

    ArrayView(const ArrayView& other);

    ArrayView(ArrayView&&) noexcept;

    // set

    void set(const Array<T, N, Handle>& other);

    void set(const ArrayView<T, N, Handle>& other);

    void set(const ArrayView& other);

    // Assignment Operators
    ArrayView& operator=(const ArrayView<T, N, Handle>& other);

    ArrayView& operator=(const ArrayView& other);

    ArrayView& operator=(ArrayView&& other) noexcept;
};

template <class T>
using ArrayView1 = ArrayView<T, 1, CpuMemory<T>>;

template <class T>
using ArrayView2 = ArrayView<T, 2, CpuMemory<T>>;

template <class T>
using ArrayView3 = ArrayView<T, 3, CpuMemory<T>>;

template <class T>
using ArrayView4 = ArrayView<T, 4, CpuMemory<T>>;

template <class T>
using ConstArrayView1 = ArrayView<const T, 1, CpuMemory<T>>;

template <class T>
using ConstArrayView2 = ArrayView<const T, 2, CpuMemory<T>>;

template <class T>
using ConstArrayView3 = ArrayView<const T, 3, CpuMemory<T>>;

template <class T>
using ConstArrayView4 = ArrayView<const T, 4, CpuMemory<T>>;

#ifdef JET_USE_CUDA

template <class T>
using NewCudaArrayView1 = ArrayView<T, 1, CudaMemory<T>>;

template <class T>
using NewCudaArrayView2 = ArrayView<T, 2, CudaMemory<T>>;

template <class T>
using NewCudaArrayView3 = ArrayView<T, 3, CudaMemory<T>>;

template <class T>
using NewCudaArrayView4 = ArrayView<T, 4, CudaMemory<T>>;

template <class T>
using NewCudaConstArrayView1 = ArrayView<const T, 1, CudaMemory<T>>;

template <class T>
using NewCudaConstArrayView2 = ArrayView<const T, 2, CudaMemory<T>>;

template <class T>
using NewCudaConstArrayView3 = ArrayView<const T, 3, CudaMemory<T>>;

template <class T>
using NewCudaConstArrayView4 = ArrayView<const T, 4, CudaMemory<T>>;

#endif

}  // namespace jet

#include <jet/detail/array_view-inl.h>

#endif  // INCLUDE_JET_ARRAY_VIEW_H_
