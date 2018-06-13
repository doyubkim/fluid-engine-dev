// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_VIEW_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_VIEW_INL_H_

#include <jet/array_view.h>

namespace jet {

// MARK: ArrayView

template <typename T, size_t N>
ArrayView<T, N>::ArrayView() : Base() {}

template <typename T, size_t N>
ArrayView<T, N>::ArrayView(T* ptr, const SizeN<N>& size_) : ArrayView() {
    Base::setPtrAndSize(ptr, size_);
}

template <typename T, size_t N>
template <size_t M>
ArrayView<T, N>::ArrayView(typename std::enable_if<(M == 1), T>::type* ptr,
                           size_t size_)
    : ArrayView(ptr, SizeN<N>{size_}) {}

template <typename T, size_t N>
ArrayView<T, N>::ArrayView(Array<T, N>& other) : ArrayView() {
    set(other);
}

template <typename T, size_t N>
ArrayView<T, N>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T, size_t N>
ArrayView<T, N>::ArrayView(ArrayView&& other) noexcept : ArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N>
void ArrayView<T, N>::set(Array<T, N>& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
void ArrayView<T, N>::set(const ArrayView& other) {
    Base::setPtrAndSize(const_cast<T*>(other.data()), other.size());
}

template <typename T, size_t N>
ArrayView<T, N>& ArrayView<T, N>::operator=(const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
ArrayView<T, N>& ArrayView<T, N>::operator=(ArrayView&& other) noexcept {
    Base::setPtrAndSize(other.data(), other.size());
    other.setPtrAndSize(nullptr, SizeN<N>{});
    return *this;
}

// MARK: ConstArrayView

template <typename T, size_t N>
ArrayView<const T, N>::ArrayView() : Base() {}

template <typename T, size_t N>
ArrayView<const T, N>::ArrayView(const T* ptr, const SizeN<N>& size_)
    : ArrayView() {
    Base::setPtrAndSize(ptr, size_);
}

template <typename T, size_t N>
template <size_t M>
ArrayView<const T, N>::ArrayView(
    const typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : ArrayView(ptr, SizeN<N>{size_}) {}

template <typename T, size_t N>
ArrayView<const T, N>::ArrayView(const Array<T, N>& other) : ArrayView() {
    set(other);
}

template <typename T, size_t N>
ArrayView<const T, N>::ArrayView(const ArrayView<T, N>& other) {
    set(other);
}

template <typename T, size_t N>
ArrayView<const T, N>::ArrayView(const ArrayView<const T, N>& other) {
    set(other);
}

template <typename T, size_t N>
ArrayView<const T, N>::ArrayView(ArrayView&& other) noexcept : ArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N>
void ArrayView<const T, N>::set(const Array<T, N>& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
void ArrayView<const T, N>::set(const ArrayView<T, N>& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
void ArrayView<const T, N>::set(const ArrayView<const T, N>& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
ArrayView<const T, N>& ArrayView<const T, N>::operator=(
    const ArrayView<T, N>& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
ArrayView<const T, N>& ArrayView<const T, N>::operator=(
    const ArrayView<const T, N>& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
ArrayView<const T, N>& ArrayView<const T, N>::operator=(
    ArrayView<const T, N>&& other) noexcept {
    Base::setPtrAndSize(other.data(), other.size());
    other.setPtrAndSize(nullptr, SizeN<N>{});
    return *this;
}

}  // namespace jet

#include <jet/detail/array_view-inl.h>

#endif  // INCLUDE_JET_DETAIL_ARRAY_VIEW_INL_H_
