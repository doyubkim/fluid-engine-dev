// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_VIEW3_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_VIEW3_INL_H_

#include <jet/array_view3.h>

#include <algorithm>

namespace jet {

template <typename T>
ArrayView<T, 3>::ArrayView() : _data(nullptr) {}

template <typename T>
ArrayView<T, 3>::ArrayView(T* data, const Size3& size) {
    set(data, size);
}

template <typename T>
ArrayView<T, 3>::ArrayView(const Array1<T>& array, const Size3& size) {
    set(array, size);
}

template <typename T>
ArrayView<T, 3>::ArrayView(const Array3<T>& array) {
    set(array);
}

template <typename T>
ArrayView<T, 3>::ArrayView(const std::vector<T>& vec, const Size3& size) {
    set(vec, size);
}

template <typename T>
ArrayView<T, 3>::ArrayView(const ArrayView<T, 1>& other, const Size3& size) {
    set(other, size);
}

template <typename T>
ArrayView<T, 3>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T>
ArrayView<T, 3>::ArrayView(ArrayView&& other) {
    (*this) = std::move(other);
}

template <typename T>
void ArrayView<T, 3>::set(T* data, const Size3& size) {
    _data = data;
    _size = size;
}

template <typename T>
void ArrayView<T, 3>::set(const Array1<T>& array, const Size3& size) {
    JET_ASSERT(array.size() == size.x * size.y * size.z);
    set(array.data(), size);
}

template <typename T>
void ArrayView<T, 3>::set(const Array3<T>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void ArrayView<T, 3>::set(const std::vector<T>& vec, const Size3& size) {
    JET_ASSERT(vec.size() == size.x * size.y * size.z);
    set(vec.data(), size);
}

template <typename T>
void ArrayView<T, 3>::set(const ArrayView<T, 1>& other, const Size3& size) {
    JET_ASSERT(other.size() == size.x * size.y * size.z);
    set(other.data(), size);
}

template <typename T>
void ArrayView<T, 3>::set(const ArrayView& other) {
    set(other.data(), other.size());
}

template <typename T>
void ArrayView<T, 3>::swap(ArrayView& other) {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
}

template <typename T>
const Size3& ArrayView<T, 3>::size() const {
    return _size;
}

template <typename T>
size_t ArrayView<T, 3>::width() const {
    return _size.x;
}

template <typename T>
size_t ArrayView<T, 3>::height() const {
    return _size.y;
}

template <typename T>
size_t ArrayView<T, 3>::depth() const {
    return _size.z;
}

template <typename T>
T* ArrayView<T, 3>::data() {
    return _data;
}

template <typename T>
const T* ArrayView<T, 3>::data() const {
    return _data;
}

template <typename T>
T* ArrayView<T, 3>::begin() {
    return _data;
}

template <typename T>
const T* ArrayView<T, 3>::begin() const {
    return _data;
}

template <typename T>
T* ArrayView<T, 3>::end() {
    return _data + _size.x * _size.y * _size.z;
}

template <typename T>
const T* ArrayView<T, 3>::end() const {
    return _data + _size.x * _size.y * _size.z;
}

template <typename T>
T& ArrayView<T, 3>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& ArrayView<T, 3>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
T& ArrayView<T, 3>::operator()(size_t i, size_t j, size_t k) {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
const T& ArrayView<T, 3>::operator()(size_t i, size_t j, size_t k) const {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
ArrayView<T, 3>& ArrayView<T, 3>::operator=(const Array3<T>& array) {
    set(array);
    return *this;
}

template <typename T>
ArrayView<T, 3>& ArrayView<T, 3>::operator=(const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T>
ArrayView<T, 3>& ArrayView<T, 3>::operator=(ArrayView&& other) {
    _data = other._data;
    _size = other._size;
    other._data = nullptr;
    other._size = Size3();
    return *this;
}

//

template <typename T>
ConstArrayView<T, 3>::ConstArrayView() : _data(nullptr) {}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(const T* data, const Size3& size) {
    set(data, size);
}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(const std::vector<T>& vec,
                                     const Size3& size) {
    set(vec, size);
}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(const Array1<T>& array,
                                     const Size3& size) {
    set(array, size);
}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(const Array3<T>& array) {
    set(array);
}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(const ArrayView<T, 1>& other,
                                     const Size3& size) {
    set(other, size);
}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(const ArrayView<T, 3>& other) {
    set(other);
}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(const ConstArrayView<T, 1>& other,
                                     const Size3& size) {
    set(other, size);
}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(const ConstArrayView& other) {
    set(other);
}

template <typename T>
ConstArrayView<T, 3>::ConstArrayView(ConstArrayView&& other) {
    (*this) = std::move(other);
}

template <typename T>
const Size3& ConstArrayView<T, 3>::size() const {
    return _size;
}

template <typename T>
const T* ConstArrayView<T, 3>::data() const {
    return _data;
}

template <typename T>
const T* ConstArrayView<T, 3>::begin() const {
    return _data;
}

template <typename T>
const T* ConstArrayView<T, 3>::end() const {
    return _data + _size.x * _size.y * _size.z;
}

template <typename T>
const T& ConstArrayView<T, 3>::operator[](size_t i) const {
    JET_ASSERT(i < _size.x * _size.y * _size.z);
    return _data[i];
}

template <typename T>
const T& ConstArrayView<T, 3>::operator()(size_t i, size_t j, size_t k) const {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
ConstArrayView<T, 3>& ConstArrayView<T, 3>::operator=(const Array3<T>& array) {
    set(array);
    return *this;
}

template <typename T>
ConstArrayView<T, 3>& ConstArrayView<T, 3>::operator=(
    const ArrayView<T, 3>& other) {
    set(other);
    return *this;
}

template <typename T>
ConstArrayView<T, 3>& ConstArrayView<T, 3>::operator=(
    const ConstArrayView& other) {
    set(other);
    return *this;
}

template <typename T>
ConstArrayView<T, 3>& ConstArrayView<T, 3>::operator=(ConstArrayView&& other) {
    _data = other._data;
    _size = other._size;
    other._data = nullptr;
    other._size = Size3();
    return *this;
}

template <typename T>
void ConstArrayView<T, 3>::set(const T* data, const Size3& size) {
    _data = data;
    _size = size;
}

template <typename T>
void ConstArrayView<T, 3>::set(const std::vector<T>& vec, const Size3& size) {
    JET_ASSERT(vec.size() == size.x * size.y * size.z);
    set(vec.data(), size);
}

template <typename T>
void ConstArrayView<T, 3>::set(const Array1<T>& array, const Size3& size) {
    JET_ASSERT(array.size() == size.x * size.y * size.z);
    set(array.data(), size);
}

template <typename T>
void ConstArrayView<T, 3>::set(const Array3<T>& array) {
    set(array.data(), array.size());
}

template <typename T>
void ConstArrayView<T, 3>::set(const ArrayView<T, 1>& other,
                               const Size3& size) {
    JET_ASSERT(other.size() == size.x * size.y * size.z);
    set(other.data(), size);
}

template <typename T>
void ConstArrayView<T, 3>::set(const ArrayView<T, 3>& other) {
    set(other.data(), other.size());
}

template <typename T>
void ConstArrayView<T, 3>::set(const ConstArrayView<T, 1>& other) {
    set(other.data(), other.size());
}

template <typename T>
void ConstArrayView<T, 3>::set(const ConstArrayView& other) {
    set(other.data(), other.size());
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_VIEW3_INL_H_
