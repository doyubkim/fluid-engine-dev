// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW2_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW2_INL_H_

#include <jet/cuda_array_view2.h>

#include <thrust/device_ptr.h>

namespace jet {

template <typename T>
CudaArrayView<T, 2>::CudaArrayView() {}

template <typename T>
CudaArrayView<T, 2>::CudaArrayView(pointer data, const Size2& size) {
    set(data, size);
}

template <typename T>
CudaArrayView<T, 2>::CudaArrayView(const CudaArray<T, 2>& array) {
    set(array);
}

template <typename T>
CudaArrayView<T, 2>::CudaArrayView(const CudaArrayView& other) {
    set(other);
}

template <typename T>
CudaArrayView<T, 2>::CudaArrayView(CudaArrayView&& other) {
    *this = std::move(other);
}

template <typename T>
void CudaArrayView<T, 2>::set(pointer data, const Size2& size) {
    _data = thrust::device_pointer_cast<T>(data);
    _size = size;
}

template <typename T>
void CudaArrayView<T, 2>::set(const CudaArray<T, 2>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void CudaArrayView<T, 2>::set(const CudaArrayView& other) {
    _data = other._data;
    _size = other._size;
}

template <typename T>
void CudaArrayView<T, 2>::swap(CudaArrayView& other) {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
}

template <typename T>
const Size2& CudaArrayView<T, 2>::size() const {
    return _size;
}

template <typename T>
size_t CudaArrayView<T, 2>::width() const {
    return _size.x;
}

template <typename T>
size_t CudaArrayView<T, 2>::height() const {
    return _size.y;
}

template <typename T>
typename CudaArrayView<T, 2>::pointer CudaArrayView<T, 2>::data() {
    return thrust::raw_pointer_cast(_data);
}

template <typename T>
typename CudaArrayView<T, 2>::const_pointer CudaArrayView<T, 2>::data() const {
    return thrust::raw_pointer_cast(_data);
}

template <typename T>
typename CudaArrayView<T, 2>::iterator CudaArrayView<T, 2>::begin() {
    return _data;
}

template <typename T>
typename CudaArrayView<T, 2>::const_iterator CudaArrayView<T, 2>::begin()
    const {
    return _data;
}

template <typename T>
typename CudaArrayView<T, 2>::iterator CudaArrayView<T, 2>::end() {
    return _data + _size.x * _size.y;
}

template <typename T>
typename CudaArrayView<T, 2>::const_iterator CudaArrayView<T, 2>::end() const {
    return _data + _size.x * _size.y;
}

template <typename T>
typename CudaArrayView<T, 2>::reference CudaArrayView<T, 2>::operator[](
    size_t i) {
    return _data[i];
}

template <typename T>
typename CudaArrayView<T, 2>::value_type CudaArrayView<T, 2>::operator[](
    size_t i) const {
    return _data[i];
}

template <typename T>
typename CudaArrayView<T, 2>::reference CudaArrayView<T, 2>::operator()(
    size_t i, size_t j) {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
typename CudaArrayView<T, 2>::value_type CudaArrayView<T, 2>::operator()(
    size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
CudaArrayView<T, 2>& CudaArrayView<T, 2>::operator=(
    const CudaArray<T, 2>& array) {
    set(array);
    return *this;
}

template <typename T>
CudaArrayView<T, 2>& CudaArrayView<T, 2>::operator=(
    const CudaArrayView& other) {
    set(other);
    return *this;
}

template <typename T>
CudaArrayView<T, 2>& CudaArrayView<T, 2>::operator=(CudaArrayView&& view) {
    _data = view._data;
    _size = view._size;
    view._data = thrust::device_ptr<T>();
    view._size = Size2{};
    return *this;
}

//

template <typename T>
ConstCudaArrayView<T, 2>::ConstCudaArrayView() {}

template <typename T>
ConstCudaArrayView<T, 2>::ConstCudaArrayView(const_pointer data,
                                             const Size2& size) {
    set(data, size);
}

template <typename T>
ConstCudaArrayView<T, 2>::ConstCudaArrayView(const CudaArray<T, 2>& array) {
    set(array);
}

template <typename T>
ConstCudaArrayView<T, 2>::ConstCudaArrayView(const CudaArrayView<T, 2>& other) {
    set(other);
}

template <typename T>
ConstCudaArrayView<T, 2>::ConstCudaArrayView(const ConstCudaArrayView& other) {
    set(other);
}

template <typename T>
ConstCudaArrayView<T, 2>::ConstCudaArrayView(ConstCudaArrayView&& other) {
    *this = std::move(other);
}

template <typename T>
const Size2& ConstCudaArrayView<T, 2>::size() const {
    return _size;
}

template <typename T>
size_t ConstCudaArrayView<T, 2>::width() const {
    return _size.x;
}

template <typename T>
size_t ConstCudaArrayView<T, 2>::height() const {
    return _size.y;
}

template <typename T>
typename ConstCudaArrayView<T, 2>::const_pointer
ConstCudaArrayView<T, 2>::data() const {
    return thrust::raw_pointer_cast(_data);
}

template <typename T>
typename ConstCudaArrayView<T, 2>::const_iterator
ConstCudaArrayView<T, 2>::begin() const {
    return _data;
}

template <typename T>
typename ConstCudaArrayView<T, 2>::const_iterator
ConstCudaArrayView<T, 2>::end() const {
    return _data + _size.x * _size.y;
}

template <typename T>
typename ConstCudaArrayView<T, 2>::value_type ConstCudaArrayView<T, 2>::
operator[](size_t i) const {
    return _data[i];
}

template <typename T>
typename ConstCudaArrayView<T, 2>::value_type ConstCudaArrayView<T, 2>::
operator()(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
ConstCudaArrayView<T, 2>& ConstCudaArrayView<T, 2>::operator=(
    const CudaArray<T, 2>& array) {
    set(array);
    return *this;
}

template <typename T>
ConstCudaArrayView<T, 2>& ConstCudaArrayView<T, 2>::operator=(
    const CudaArrayView<T, 2>& other) {
    set(other);
    return *this;
}

template <typename T>
ConstCudaArrayView<T, 2>& ConstCudaArrayView<T, 2>::operator=(
    const ConstCudaArrayView& other) {
    set(other);
    return *this;
}

template <typename T>
ConstCudaArrayView<T, 2>& ConstCudaArrayView<T, 2>::operator=(
    ConstCudaArrayView&& view) {
    _data = view._data;
    _size = view._size;
    view._data = thrust::device_ptr<T>();
    view._size = Size2{};
    return *this;
}

template <typename T>
void ConstCudaArrayView<T, 2>::set(const_pointer data, const Size2& size) {
    _data = thrust::device_pointer_cast<T>(const_cast<T*>(data));
    _size = size;
}

template <typename T>
void ConstCudaArrayView<T, 2>::set(const CudaArray<T, 2>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void ConstCudaArrayView<T, 2>::set(const CudaArrayView<T, 2>& other) {
    _data = other.data();
    _size = other.size();
}

template <typename T>
void ConstCudaArrayView<T, 2>::set(const ConstCudaArrayView& other) {
    _data = other._data;
    _size = other._size;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW2_INL_H_

#endif  // JET_USE_CUDA
