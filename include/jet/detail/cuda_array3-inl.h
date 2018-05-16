// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY3_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY3_INL_H_

#include <jet/cuda_array3.h>
#include <jet/cuda_array_view3.h>

namespace jet {

template <typename T>
CudaArray<T, 3>::CudaArray() {}

template <typename T>
CudaArray<T, 3>::CudaArray(const Size3& size, const T& initVal) {
    resize(size, initVal);
}

template <typename T>
CudaArray<T, 3>::CudaArray(size_t width, size_t height, size_t depth,
                           const T& initVal) {
    resize({width, height, depth}, initVal);
}

template <typename T>
CudaArray<T, 3>::CudaArray(const ConstArrayView<T, 3>& view) {
    set(view);
}

template <typename T>
CudaArray<T, 3>::CudaArray(const ConstCudaArrayView<T, 3>& view) {
    set(view);
}

template <typename T>
CudaArray<T, 3>::CudaArray(
    const std::initializer_list<
        std::initializer_list<std::initializer_list<T>>>& lst) {
    set(lst);
}

template <typename T>
CudaArray<T, 3>::CudaArray(const CudaArray& other) {
    set(other);
}

template <typename T>
CudaArray<T, 3>::CudaArray(CudaArray&& other) {
    (*this) = std::move(other);
}

template <typename T>
void CudaArray<T, 3>::set(const T& value) {
    thrust::fill(_data.begin(), _data.end(), value);
}

template <typename T>
void CudaArray<T, 3>::set(const ConstArrayView<T, 3>& view) {
    _size = view.size();
    _data.resize(_size.x * _size.y * _size.z);
    thrust::copy(view.begin(), view.end(), _data.begin());
}

template <typename T>
void CudaArray<T, 3>::set(const ConstCudaArrayView<T, 3>& view) {
    _size = view.size();
    _data.resize(_size.x * _size.y * _size.z);
    thrust::copy(view.begin(), view.end(), _data.begin());
}

template <typename T>
void CudaArray<T, 3>::set(
    const std::initializer_list<
        std::initializer_list<std::initializer_list<T>>>& lst) {
    Array3<T> temp(lst);
    set(temp.view());
}

template <typename T>
void CudaArray<T, 3>::set(const CudaArray& other) {
    _size = other._size;
    _data = other._data;
}

template <typename T>
void CudaArray<T, 3>::clear() {
    _size = Size3{};
    _data.clear();
}

template <typename T>
void CudaArray<T, 3>::resize(const Size3& size, const T& initVal) {
    _size = size;
    _data.resize(_size.x * _size.y * _size.z, initVal);

    // TODO: Should behave like Array<T, 3>
}

template <typename T>
void CudaArray<T, 3>::swap(CudaArray& other) {
    std::swap(other._data, _data);
    std::swap(other._size, _size);
}

template <typename T>
const Size3& CudaArray<T, 3>::size() const {
    return _size;
}

template <typename T>
size_t CudaArray<T, 3>::width() const {
    return _size.x;
}

template <typename T>
size_t CudaArray<T, 3>::height() const {
    return _size.y;
}

template <typename T>
size_t CudaArray<T, 3>::depth() const {
    return _size.z;
}

template <typename T>
typename CudaArray<T, 3>::pointer CudaArray<T, 3>::data() {
    return thrust::raw_pointer_cast(_data.data());
}

template <typename T>
typename CudaArray<T, 3>::const_pointer CudaArray<T, 3>::data() const {
    return thrust::raw_pointer_cast(_data.data());
}

template <typename T>
typename CudaArray<T, 3>::iterator CudaArray<T, 3>::begin() {
    return _data.data();
}

template <typename T>
typename CudaArray<T, 3>::iterator CudaArray<T, 3>::begin() const {
    return _data.data();
}

template <typename T>
typename CudaArray<T, 3>::iterator CudaArray<T, 3>::end() {
    return _data.data() + _data.size();
}

template <typename T>
typename CudaArray<T, 3>::iterator CudaArray<T, 3>::end() const {
    return _data.data() + _data.size();
}

template <typename T>
CudaArrayView<T, 3> CudaArray<T, 3>::view() {
    return CudaArrayView<T, 3>(*this);
}

template <typename T>
ConstCudaArrayView<T, 3> CudaArray<T, 3>::view() const {
    return CudaArrayView<T, 3>(*this);
}

template <typename T>
typename CudaArray<T, 3>::reference CudaArray<T, 3>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
typename CudaArray<T, 3>::value_type CudaArray<T, 3>::operator[](
    size_t i) const {
    return _data[i];
}

template <typename T>
typename CudaArray<T, 3>::reference CudaArray<T, 3>::operator()(size_t i,
                                                                size_t j,
                                                                size_t k) {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
typename CudaArray<T, 3>::value_type CudaArray<T, 3>::operator()(
    size_t i, size_t j, size_t k) const {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
CudaArray<T, 3>& CudaArray<T, 3>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
CudaArray<T, 3>& CudaArray<T, 3>::operator=(const ArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
CudaArray<T, 3>& CudaArray<T, 3>::operator=(const CudaArrayView<T, 3>& view) {
    set(view);
    return *this;
}

template <typename T>
CudaArray<T, 3>& CudaArray<T, 3>::operator=(
    const std::initializer_list<
        std::initializer_list<std::initializer_list<T>>>& lst) {
    set(lst);
    return *this;
}

template <typename T>
CudaArray<T, 3>& CudaArray<T, 3>::operator=(const CudaArray& other) {
    set(other);
    return *this;
}

template <typename T>
CudaArray<T, 3>& CudaArray<T, 3>::operator=(CudaArray&& other) {
    swap(other);
    other.clear();
    return *this;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY3_INL_H_

#endif  // JET_USE_CUDA
