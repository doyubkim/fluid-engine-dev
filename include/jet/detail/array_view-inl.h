// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_VIEW_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_VIEW_INL_H_

#include <jet/array.h>
#include <jet/array_view.h>

namespace jet {

// MARK: ArrayView

template <typename T, size_t N, typename Device>
ArrayView<T, N, Device>::ArrayView() : Base() {}

template <typename T, size_t N, typename Device>
ArrayView<T, N, Device>::ArrayView(T* ptr, const Vector<size_t, N>& size_)
    : ArrayView() {
    Base::setHandleAndSize(MemoryHandle(ptr), size_);
}

template <typename T, size_t N, typename Device>
template <size_t M>
ArrayView<T, N, Device>::ArrayView(
    typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : ArrayView(ptr, Vector<size_t, N>{size_}) {}

template <typename T, size_t N, typename Device>
ArrayView<T, N, Device>::ArrayView(Array<T, N, Device>& other) : ArrayView() {
    set(other);
}

template <typename T, size_t N, typename Device>
ArrayView<T, N, Device>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T, size_t N, typename Device>
ArrayView<T, N, Device>::ArrayView(ArrayView&& other) noexcept : ArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N, typename Device>
void ArrayView<T, N, Device>::set(Array<T, N, Device>& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N, typename Device>
void ArrayView<T, N, Device>::set(const ArrayView& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N, typename Device>
void ArrayView<T, N, Device>::fill(const T& val) {
    Device::fill(*this, val);
}

template <typename T, size_t N, typename Device>
ArrayView<T, N, Device>& ArrayView<T, N, Device>::operator=(
    const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N, typename Device>
ArrayView<T, N, Device>& ArrayView<T, N, Device>::operator=(
    ArrayView&& other) noexcept {
    Base::setHandleAndSize(other.data(), other.size());
    other.setHandleAndSize(MemoryHandle(), Vector<size_t, N>{});
    return *this;
}

// MARK: ConstArrayView

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>::ArrayView() : Base() {}

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>::ArrayView(const T* ptr,
                                         const Vector<size_t, N>& size_)
    : ArrayView() {
    Base::setHandleAndSize(MemoryHandle(ptr), size_);
}

template <typename T, size_t N, typename Device>
template <size_t M>
ArrayView<const T, N, Device>::ArrayView(
    const typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : ArrayView(ptr, Vector<size_t, N>{size_}) {}

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>::ArrayView(const Array<T, N, Device>& other)
    : ArrayView() {
    set(other);
}

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>::ArrayView(const ArrayView<T, N, Device>& other) {
    set(other);
}

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>::ArrayView(ArrayView&& other) noexcept
    : ArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N, typename Device>
void ArrayView<const T, N, Device>::set(const Array<T, N, Device>& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N, typename Device>
void ArrayView<const T, N, Device>::set(const ArrayView<T, N, Device>& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N, typename Device>
void ArrayView<const T, N, Device>::set(const ArrayView& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>& ArrayView<const T, N, Device>::operator=(
    const ArrayView<T, N, Device>& other) {
    set(other);
    return *this;
}

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>& ArrayView<const T, N, Device>::operator=(
    const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device>& ArrayView<const T, N, Device>::operator=(
    ArrayView&& other) noexcept {
    Base::setHandleAndSize(other.data(), other.size());
    other.setHandleAndSize(MemoryHandle(), Vector<size_t, N>{});
    return *this;
}

}  // namespace jet

#include <jet/detail/array_view-inl.h>

#endif  // INCLUDE_JET_DETAIL_ARRAY_VIEW_INL_H_
