// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY2_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY2_INL_H_

#include <jet/cuda_array2.h>
#include <jet/cuda_array_view2.h>

namespace jet {

template <typename T>
CudaArray<T, 2>::CudaArray() {}

template <typename T>
CudaArray<T, 2>::CudaArray(const Size2& size, const T& initVal) {
    resize(size, initVal);
}

template <typename T>
CudaArray<T, 2>::CudaArray(size_t width, size_t height, const T& initVal) {
    resize({width, height}, initVal);
}

template <typename T>
CudaArray<T, 2>::CudaArray(const ConstArrayView<T, 2>& view) {
    set(view);
}

template <typename T>
CudaArray<T, 2>::CudaArray(const ConstCudaArrayView<T, 2>& view) {
    set(view);
}

template <typename T>
CudaArray<T, 2>::CudaArray(
    const std::initializer_list<std::initializer_list<T>>& lst) {
    set(lst);
}

template <typename T>
CudaArray<T, 2>::CudaArray(const CudaArray& other) {
    set(other);
}

template <typename T>
CudaArray<T, 2>::CudaArray(CudaArray&& other) {
    (*this) = std::move(other);
}

template <typename T>
void CudaArray<T, 2>::set(const T& value) {
    thrust::fill(_data.begin(), _data.end(), value);
}

template <typename T>
void CudaArray<T, 2>::set(const ConstArrayView<T, 2>& view) {
    _size = view.size();
    _data.resize(_size.x * _size.y);
    thrust::copy(view.begin(), view.end(), _data.begin());
}

template <typename T>
void CudaArray<T, 2>::set(const ConstCudaArrayView<T, 2>& view) {
    _size = view.size();
    _data.resize(_size.x * _size.y);
    thrust::copy(view.begin(), view.end(), _data.begin());
}

template <typename T>
void CudaArray<T, 2>::set(
    const std::initializer_list<std::initializer_list<T>>& lst) {
    Array2<T> temp(lst);
    set(temp.view());
}

template <typename T>
void CudaArray<T, 2>::set(const CudaArray& other) {
    _size = other._size;
    _data = other._data;
}

template <typename T>
void CudaArray<T, 2>::clear() {
    _size = Size2{};
    _data.clear();
}

template <typename T>
void CudaArray<T, 2>::resize(const Size2& size, const T& initVal) {
    _size = size;
    _data.resize(_size.x * _size.y, initVal);

    // TODO: Should behave like Array<T, 2>
}

template <typename T>
void CudaArray<T, 2>::swap(CudaArray& other) {
    std::swap(other._data, _data);
    std::swap(other._size, _size);
}

template <typename T>
const Size2& CudaArray<T, 2>::size() const {
    return _size;
}

template <typename T>
size_t CudaArray<T, 2>::width() const {
    return _size.x;
}

template <typename T>
size_t CudaArray<T, 2>::height() const {
    return _size.y;
}

template <typename T>
typename CudaArray<T, 2>::pointer CudaArray<T, 2>::data() {
    return thrust::raw_pointer_cast(_data.data());
}

template <typename T>
typename CudaArray<T, 2>::const_pointer CudaArray<T, 2>::data() const {
    return thrust::raw_pointer_cast(_data.data());
}

template <typename T>
typename CudaArray<T, 2>::iterator CudaArray<T, 2>::begin() {
    return _data.data();
}

template <typename T>
typename CudaArray<T, 2>::iterator CudaArray<T, 2>::begin() const {
    return _data.data();
}

template <typename T>
typename CudaArray<T, 2>::iterator CudaArray<T, 2>::end() {
    return _data.data() + _data.size();
}

template <typename T>
typename CudaArray<T, 2>::iterator CudaArray<T, 2>::end() const {
    return _data.data() + _data.size();
}

template <typename T>
CudaArrayView<T, 2> CudaArray<T, 2>::view() {
    return CudaArrayView<T, 2>(*this);
}

template <typename T>
ConstCudaArrayView<T, 2> CudaArray<T, 2>::view() const {
    return CudaArrayView<T, 2>(*this);
}

template <typename T>
typename CudaArray<T, 2>::reference CudaArray<T, 2>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
typename CudaArray<T, 2>::value_type CudaArray<T, 2>::operator[](
    size_t i) const {
    return _data[i];
}

template <typename T>
typename CudaArray<T, 2>::reference CudaArray<T, 2>::operator()(size_t i,
                                                                size_t j) {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
typename CudaArray<T, 2>::value_type CudaArray<T, 2>::operator()(
    size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
CudaArray<T, 2>& CudaArray<T, 2>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
CudaArray<T, 2>& CudaArray<T, 2>::operator=(const ArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
CudaArray<T, 2>& CudaArray<T, 2>::operator=(const CudaArrayView<T, 2>& view) {
    set(view);
    return *this;
}

template <typename T>
CudaArray<T, 2>& CudaArray<T, 2>::operator=(
    const std::initializer_list<std::initializer_list<T>>& lst) {
    set(lst);
    return *this;
}

template <typename T>
CudaArray<T, 2>& CudaArray<T, 2>::operator=(const CudaArray& other) {
    set(other);
    return *this;
}

template <typename T>
CudaArray<T, 2>& CudaArray<T, 2>::operator=(CudaArray&& other) {
    swap(other);
    other.clear();
    return *this;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY2_INL_H_

#endif  // JET_USE_CUDA
