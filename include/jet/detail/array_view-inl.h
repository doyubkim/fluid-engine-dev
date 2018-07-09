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

template <typename T, size_t N, typename Handle>
ArrayView<T, N, Handle>::ArrayView() : Base() {}

template <typename T, size_t N, typename Handle>
ArrayView<T, N, Handle>::ArrayView(T* ptr, const Vector<size_t, N>& size_)
    : ArrayView() {
    Base::setHandleAndSize(Handle(ptr), size_);
}

template <typename T, size_t N, typename Handle>
template <size_t M>
ArrayView<T, N, Handle>::ArrayView(
    typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : ArrayView(ptr, Vector<size_t, N>{size_}) {}

template <typename T, size_t N, typename Handle>
ArrayView<T, N, Handle>::ArrayView(Array<T, N, Handle>& other) : ArrayView() {
    set(other);
}

template <typename T, size_t N, typename Handle>
ArrayView<T, N, Handle>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T, size_t N, typename Handle>
ArrayView<T, N, Handle>::ArrayView(ArrayView&& other) noexcept : ArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N, typename Handle>
void ArrayView<T, N, Handle>::set(Array<T, N, Handle>& other) {
    Base::setHandleAndSize(other.devicePtr(), other.size());
}

template <typename T, size_t N, typename Handle>
void ArrayView<T, N, Handle>::set(const ArrayView& other) {
    Base::setHandleAndSize(other.devicePtr(), other.size());
}

template <typename T, size_t N, typename Handle>
void ArrayView<T, N, Handle>::fill(const T& val) {
#ifdef JET_USE_CUDA
    thrust::fill(begin(), end(), val);
#else
    std::fill(begin(), end(), val);
#endif
}

template <typename T, size_t N, typename Handle>
ArrayView<T, N, Handle>& ArrayView<T, N, Handle>::operator=(
    const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N, typename Handle>
ArrayView<T, N, Handle>& ArrayView<T, N, Handle>::operator=(
    ArrayView&& other) noexcept {
    Base::setHandleAndSize(other.data(), other.size());
    other.setHandleAndSize(Handle(), Vector<size_t, N>{});
    return *this;
}

// MARK: ConstArrayView

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>::ArrayView() : Base() {}

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>::ArrayView(const T* ptr,
                                         const Vector<size_t, N>& size_)
    : ArrayView() {
    Base::setHandleAndSize(Handle(ptr), size_);
}

template <typename T, size_t N, typename Handle>
template <size_t M>
ArrayView<const T, N, Handle>::ArrayView(
    const typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : ArrayView(ptr, Vector<size_t, N>{size_}) {}

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>::ArrayView(const Array<T, N, Handle>& other)
    : ArrayView() {
    set(other);
}

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>::ArrayView(const ArrayView<T, N, Handle>& other) {
    set(other);
}

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>::ArrayView(ArrayView&& other) noexcept
    : ArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N, typename Handle>
void ArrayView<const T, N, Handle>::set(const Array<T, N, Handle>& other) {
    Base::setHandleAndSize(other.devicePtr(), other.size());
}

template <typename T, size_t N, typename Handle>
void ArrayView<const T, N, Handle>::set(const ArrayView<T, N, Handle>& other) {
    Base::setHandleAndSize(other.devicePtr(), other.size());
}

template <typename T, size_t N, typename Handle>
void ArrayView<const T, N, Handle>::set(const ArrayView& other) {
    Base::setHandleAndSize(other.devicePtr(), other.size());
}

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>& ArrayView<const T, N, Handle>::operator=(
    const ArrayView<T, N, Handle>& other) {
    set(other);
    return *this;
}

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>& ArrayView<const T, N, Handle>::operator=(
    const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle>& ArrayView<const T, N, Handle>::operator=(
    ArrayView&& other) noexcept {
    Base::setHandleAndSize(other.data(), other.size());
    other.setHandleAndSize(Handle(), Vector<size_t, N>{});
    return *this;
}

}  // namespace jet

#include <jet/detail/array_view-inl.h>

#endif  // INCLUDE_JET_DETAIL_ARRAY_VIEW_INL_H_
