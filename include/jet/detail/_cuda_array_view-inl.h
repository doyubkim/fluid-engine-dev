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
// MARK: CudaArrayView

template <typename T, size_t N>
CudaArrayView<T, N>::CudaArrayView() : Base() {}

template <typename T, size_t N>
CudaArrayView<T, N>::CudaArrayView(T* ptr, const CudaStdArray<size_t, N>& size_)
    : CudaArrayView() {
    Base::setPtrAndSize(ptr, size_);
}

template <typename T, size_t N>
template <size_t M>
CudaArrayView<T, N>::CudaArrayView(
    typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : CudaArrayView(ptr, CudaStdArray<size_t, N>{size_}) {}

template <typename T, size_t N>
CudaArrayView<T, N>::CudaArrayView(CudaArray<T, N>& other) : CudaArrayView() {
    set(other);
}

template <typename T, size_t N>
CudaArrayView<T, N>::CudaArrayView(const CudaArrayView& other) {
    set(other);
}

template <typename T, size_t N>
CudaArrayView<T, N>::CudaArrayView(CudaArrayView&& other) noexcept
    : CudaArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N>
void CudaArrayView<T, N>::set(CudaArray<T, N>& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
void CudaArrayView<T, N>::set(const CudaArrayView& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
void CudaArrayView<T, N>::fill(const T& val) {
    cudaFill(data(), val);
}

template <typename T, size_t N>
CudaArrayView<T, N>& CudaArrayView<T, N>::operator=(
    const CudaArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
CudaArrayView<T, N>& CudaArrayView<T, N>::operator=(
    CudaArrayView&& other) noexcept {
    Base::setPtrAndSize(other.data(), other.size());
    other.setPtrAndSize(nullptr, CudaStdArray<size_t, N>{});
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
// MARK: ConstCudaArrayView

template <typename T, size_t N>
CudaArrayView<const T, N>::CudaArrayView() : Base() {}

template <typename T, size_t N>
CudaArrayView<const T, N>::CudaArrayView(const T* ptr,
                                         const CudaStdArray<size_t, N>& size_)
    : CudaArrayView() {
    Base::setPtrAndSize(MemoryHandle(ptr), size_);
}

template <typename T, size_t N>
template <size_t M>
CudaArrayView<const T, N>::CudaArrayView(
    const typename std::enable_if<(M == 1), T>::type* ptr, size_t size_)
    : CudaArrayView(ptr, CudaStdArray<size_t, N>{size_}) {}

template <typename T, size_t N>
CudaArrayView<const T, N>::CudaArrayView(const CudaArray<T, N>& other)
    : CudaArrayView() {
    set(other);
}

template <typename T, size_t N>
CudaArrayView<const T, N>::CudaArrayView(const CudaArrayView<T, N>& other) {
    set(other);
}

template <typename T, size_t N>
CudaArrayView<const T, N>::CudaArrayView(const CudaArrayView& other) {
    set(other);
}

template <typename T, size_t N>
CudaArrayView<const T, N>::CudaArrayView(CudaArrayView&& other) noexcept
    : CudaArrayView() {
    *this = std::move(other);
}

template <typename T, size_t N>
void CudaArrayView<const T, N>::set(const CudaArray<T, N>& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
void CudaArrayView<const T, N>::set(const CudaArrayView<T, N>& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
void CudaArrayView<const T, N>::set(const CudaArrayView& other) {
    Base::setPtrAndSize(other.data(), other.size());
}

template <typename T, size_t N>
CudaArrayView<const T, N>& CudaArrayView<const T, N>::operator=(
    const CudaArrayView<T, N>& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
CudaArrayView<const T, N>& CudaArrayView<const T, N>::operator=(
    const CudaArrayView& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
CudaArrayView<const T, N>& CudaArrayView<const T, N>::operator=(
    CudaArrayView&& other) noexcept {
    Base::setPtrAndSize(other.data(), other.size());
    other.setPtrAndSize(nullptr, CudaStdArray<size_t, N>{});
    return *this;
}

}  // namespace jet

#include <jet/detail/array_view-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW_INL_H_
