// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW_INL_H_

#ifdef JET_USE_CUDA

#include <jet/_cuda_array.h>
#include <jet/_cuda_array_view.h>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: ArrayView

template <typename T, size_t N>
ArrayView<T, N, CudaDevice<T>>::ArrayView() : Base() {}

template <typename T, size_t N>
ArrayView<T, N, CudaDevice<T>>::ArrayView(T* ptr,
                                          const thrust::array<size_t, N>& size_)
    : ArrayView() {
    Base::setHandleAndSize(MemoryHandle(ptr), size_);
}

template <typename T, size_t N>
template <size_t M>
ArrayView<T, N, CudaDevice<T>>::ArrayView(
    typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : ArrayView(ptr, thrust::array<size_t, N>{size_}) {}

template <typename T, size_t N>
ArrayView<T, N, CudaDevice<T>>::ArrayView(Array<T, N, CudaDevice<T>>& other)
    : ArrayView() {
    set(other);
}

template <typename T, size_t N>
ArrayView<T, N, CudaDevice<T>>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T, size_t N>
ArrayView<T, N, CudaDevice<T>>::ArrayView(ArrayView&& other) noexcept
    : ArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N>
void ArrayView<T, N, CudaDevice<T>>::set(Array<T, N, CudaDevice<T>>& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N>
void ArrayView<T, N, CudaDevice<T>>::set(const ArrayView& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N>
void ArrayView<T, N, CudaDevice<T>>::fill(const T& val) {
    Device::fill(*this, val);
}

template <typename T, size_t N>
ArrayView<T, N, CudaDevice<T>>& ArrayView<T, N, CudaDevice<T>>::operator=(
    const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
ArrayView<T, N, CudaDevice<T>>& ArrayView<T, N, CudaDevice<T>>::operator=(
    ArrayView&& other) noexcept {
    Base::setHandleAndSize(other.data(), other.size());
    other.setHandleAndSize(MemoryHandle(), thrust::array<size_t, N>{});
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
// MARK: ConstArrayView

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>::ArrayView() : Base() {}

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>::ArrayView(
    const T* ptr, const thrust::array<size_t, N>& size_)
    : ArrayView() {
    Base::setHandleAndSize(MemoryHandle(ptr), size_);
}

template <typename T, size_t N>
template <size_t M>
ArrayView<const T, N, CudaDevice<T>>::ArrayView(
    const typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : ArrayView(ptr, thrust::array<size_t, N>{size_}) {}

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>::ArrayView(
    const Array<T, N, CudaDevice<T>>& other)
    : ArrayView() {
    set(other);
}

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>::ArrayView(
    const ArrayView<T, N, CudaDevice<T>>& other) {
    set(other);
}

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>::ArrayView(ArrayView&& other) noexcept
    : ArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N>
void ArrayView<const T, N, CudaDevice<T>>::set(
    const Array<T, N, CudaDevice<T>>& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N>
void ArrayView<const T, N, CudaDevice<T>>::set(
    const ArrayView<T, N, CudaDevice<T>>& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N>
void ArrayView<const T, N, CudaDevice<T>>::set(const ArrayView& other) {
    Base::setHandleAndSize(other.handle(), other.size());
}

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>& ArrayView<const T, N, CudaDevice<T>>::
operator=(const ArrayView<T, N, CudaDevice<T>>& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>& ArrayView<const T, N, CudaDevice<T>>::
operator=(const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>>& ArrayView<const T, N, CudaDevice<T>>::
operator=(ArrayView&& other) noexcept {
    Base::setHandleAndSize(other.data(), other.size());
    other.setHandleAndSize(MemoryHandle(), thrust::array<size_t, N>{});
    return *this;
}

}  // namespace jet

#include <jet/detail/array_view-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW_INL_H_
