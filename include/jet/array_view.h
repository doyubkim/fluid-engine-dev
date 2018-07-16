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
struct CpuDevice;

template <typename T, size_t N, typename Device, typename Derived>
class ArrayBase;

template <typename T, size_t N, typename Device>
class Array;

// MARK: ArrayView

template <typename T, size_t N, typename Device>
class ArrayView final
    : public ArrayBase<T, N, Device, ArrayView<T, N, Device>> {
    using Base = ArrayBase<T, N, Device, ArrayView<T, N, Device>>;
    using Base::_size;
    using Base::at;
    using Base::setHandleAndSize;

 public:
    // CTOR
    ArrayView();

    ArrayView(T* ptr, const Vector<size_t, N>& size_);

    template <size_t M = N>
    ArrayView(typename std::enable_if<(M == 1), T>::type* ptr, size_t size_);

    ArrayView(Array<T, N, Device>& other);

    ArrayView(const ArrayView& other);

    ArrayView(ArrayView&& other) noexcept;

    // set

    void set(Array<T, N, Device>& other);

    void set(const ArrayView& other);

    void fill(const T& val);

    // Assignment Operators
    ArrayView& operator=(const ArrayView& other);

    ArrayView& operator=(ArrayView&& other) noexcept;
};

template <typename T, size_t N, typename Device>
class ArrayView<const T, N, Device> final
    : public ArrayBase<const T, N, Device, ArrayView<const T, N, Device>> {
    using Base = ArrayBase<const T, N, Device, ArrayView<const T, N, Device>>;
    using Base::_size;
    using Base::setHandleAndSize;

 public:
    // CTOR
    ArrayView();

    ArrayView(const T* ptr, const Vector<size_t, N>& size_);

    template <size_t M = N>
    ArrayView(const typename std::enable_if<(M == 1), T>::type* ptr,
              size_t size_);

    ArrayView(const Array<T, N, Device>& other);

    ArrayView(const ArrayView<T, N, Device>& other);

    ArrayView(const ArrayView& other);

    ArrayView(ArrayView&&) noexcept;

    // set

    void set(const Array<T, N, Device>& other);

    void set(const ArrayView<T, N, Device>& other);

    void set(const ArrayView& other);

    // Assignment Operators
    ArrayView& operator=(const ArrayView<T, N, Device>& other);

    ArrayView& operator=(const ArrayView& other);

    ArrayView& operator=(ArrayView&& other) noexcept;
};

template <class T>
using ArrayView1 = ArrayView<T, 1, CpuDevice<T>>;

template <class T>
using ArrayView2 = ArrayView<T, 2, CpuDevice<T>>;

template <class T>
using ArrayView3 = ArrayView<T, 3, CpuDevice<T>>;

template <class T>
using ArrayView4 = ArrayView<T, 4, CpuDevice<T>>;

template <class T>
using ConstArrayView1 = ArrayView<const T, 1, CpuDevice<T>>;

template <class T>
using ConstArrayView2 = ArrayView<const T, 2, CpuDevice<T>>;

template <class T>
using ConstArrayView3 = ArrayView<const T, 3, CpuDevice<T>>;

template <class T>
using ConstArrayView4 = ArrayView<const T, 4, CpuDevice<T>>;

}  // namespace jet

#include <jet/detail/array_view-inl.h>

#endif  // INCLUDE_JET_ARRAY_VIEW_H_
