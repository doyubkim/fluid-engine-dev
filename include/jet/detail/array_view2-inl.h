// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_VIEW2_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_VIEW2_INL_H_

#include <jet/array_view2.h>

#include <algorithm>

namespace jet {

template <typename T>
ArrayView<T, 2>::ArrayView() : _data(nullptr) {}

template <typename T>
ArrayView<T, 2>::ArrayView(T* data, const Size2& size) {
    set(data, size);
}

template <typename T>
ArrayView<T, 2>::ArrayView(const Array<T, 1>& array, const Size2& size) {
    set(array, size);
}

template <typename T>
ArrayView<T, 2>::ArrayView(const Array<T, 2>& array) {
    set(array);
}

template <typename T>
ArrayView<T, 2>::ArrayView(const std::vector<T>& vec, const Size2& size) {
    set(vec, size);
}

template <typename T>
ArrayView<T, 2>::ArrayView(const ArrayView<T, 1>& other, const Size2& size) {
    set(other, size);
}

template <typename T>
ArrayView<T, 2>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T>
ArrayView<T, 2>::ArrayView(ArrayView&& other) {
    (*this) = std::move(other);
}

template <typename T>
void ArrayView<T, 2>::set(T* data, const Size2& size) {
    _data = data;
    _size = size;
}

template <typename T>
void ArrayView<T, 2>::set(const Array<T, 1>& array, const Size2& size) {
    JET_ASSERT(array.size() == size.x * size.y);
    set(array.data(), size);
}

template <typename T>
void ArrayView<T, 2>::set(const Array<T, 2>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void ArrayView<T, 2>::set(const std::vector<T>& vec, const Size2& size) {
    JET_ASSERT(vec.size() == size.x * size.y);
    set(vec.data(), size);
}

template <typename T>
void ArrayView<T, 2>::set(const ArrayView<T, 1>& other, const Size2& size) {
    JET_ASSERT(other.size() == size.x * size.y);
    set(other.data(), size);
}

template <typename T>
void ArrayView<T, 2>::set(const ArrayView& other) {
    set(other.data(), other.size());
}

template <typename T>
void ArrayView<T, 2>::swap(ArrayView& other) {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
}

template <typename T>
const Size2& ArrayView<T, 2>::size() const {
    return _size;
}

template <typename T>
size_t ArrayView<T, 2>::width() const {
    return _size.x;
}

template <typename T>
size_t ArrayView<T, 2>::height() const {
    return _size.y;
}

template <typename T>
T* ArrayView<T, 2>::data() {
    return _data;
}

template <typename T>
const T* ArrayView<T, 2>::data() const {
    return _data;
}

template <typename T>
T* ArrayView<T, 2>::begin() {
    return _data;
}

template <typename T>
const T* ArrayView<T, 2>::begin() const {
    return _data;
}

template <typename T>
T* ArrayView<T, 2>::end() {
    return _data + _size.x * _size.y;
}

template <typename T>
const T* ArrayView<T, 2>::end() const {
    return _data + _size.x * _size.y;
}

template <typename T>
T& ArrayView<T, 2>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& ArrayView<T, 2>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
T& ArrayView<T, 2>::operator()(size_t i, size_t j) {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
const T& ArrayView<T, 2>::operator()(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
ArrayView<T, 2>& ArrayView<T, 2>::operator=(const Array<T, 2>& array) {
    set(array);
    return *this;
}

template <typename T>
ArrayView<T, 2>& ArrayView<T, 2>::operator=(const ArrayView& other) {
    set(other);
    return *this;
}

template <typename T>
ArrayView<T, 2>& ArrayView<T, 2>::operator=(ArrayView&& other) {
    _data = other._data;
    _size = other._size;
    other._data = nullptr;
    other._size = Size2();
    return *this;
}

//

template <typename T>
ConstArrayView<T, 2>::ConstArrayView() : _data(nullptr) {}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(const T* data, const Size2& size) {
    set(data, size);
}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(const std::vector<T>& vec,
                                     const Size2& size) {
    set(vec, size);
}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(const Array<T, 1>& array,
                                     const Size2& size) {
    set(array, size);
}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(const Array<T, 2>& array) {
    set(array);
}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(const ArrayView<T, 1>& other,
                                     const Size2& size) {
    set(other, size);
}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(const ArrayView<T, 2>& other) {
    set(other);
}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(const ConstArrayView<T, 1>& other,
                                     const Size2& size) {
    set(other, size);
}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(const ConstArrayView& other) {
    set(other);
}

template <typename T>
ConstArrayView<T, 2>::ConstArrayView(ConstArrayView&& other) {
    (*this) = std::move(other);
}

template <typename T>
const Size2& ConstArrayView<T, 2>::size() const {
    return _size;
}

template <typename T>
const T* ConstArrayView<T, 2>::data() const {
    return _data;
}

template <typename T>
const T* ConstArrayView<T, 2>::begin() const {
    return _data;
}

template <typename T>
const T* ConstArrayView<T, 2>::end() const {
    return _data + _size.x * _size.y;
}

template <typename T>
const T& ConstArrayView<T, 2>::operator[](size_t i) const {
    JET_ASSERT(i < _size.x * _size.y);
    return _data[i];
}

template <typename T>
const T& ConstArrayView<T, 2>::operator()(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
ConstArrayView<T, 2>& ConstArrayView<T, 2>::operator=(
    const Array<T, 2>& array) {
    set(array);
    return *this;
}

template <typename T>
ConstArrayView<T, 2>& ConstArrayView<T, 2>::operator=(
    const ArrayView<T, 2>& other) {
    set(other);
    return *this;
}

template <typename T>
ConstArrayView<T, 2>& ConstArrayView<T, 2>::operator=(
    const ConstArrayView& other) {
    set(other);
    return *this;
}

template <typename T>
ConstArrayView<T, 2>& ConstArrayView<T, 2>::operator=(ConstArrayView&& other) {
    _data = other._data;
    _size = other._size;
    other._data = nullptr;
    other._size = Size2();
    return *this;
}

template <typename T>
void ConstArrayView<T, 2>::set(const T* data, const Size2& size) {
    _data = data;
    _size = size;
}

template <typename T>
void ConstArrayView<T, 2>::set(const std::vector<T>& vec, const Size2& size) {
    JET_ASSERT(vec.size() == size.x * size.y);
    set(vec.data(), size);
}

template <typename T>
void ConstArrayView<T, 2>::set(const Array<T, 1>& array, const Size2& size) {
    JET_ASSERT(array.size() == size.x * size.y);
    set(array.data(), size);
}

template <typename T>
void ConstArrayView<T, 2>::set(const Array<T, 2>& array) {
    set(array.data(), array.size());
}

template <typename T>
void ConstArrayView<T, 2>::set(const ArrayView<T, 1>& other,
                               const Size2& size) {
    JET_ASSERT(other.size() == size.x * size.y);
    set(other.data(), size);
}

template <typename T>
void ConstArrayView<T, 2>::set(const ArrayView<T, 2>& other) {
    set(other.data(), other.size());
}

template <typename T>
void ConstArrayView<T, 2>::set(const ConstArrayView<T, 1>& other,
                               const Size2& size) {
    JET_ASSERT(other.size() == size.x * size.y);
    set(other.data(), size);
}

template <typename T>
void ConstArrayView<T, 2>::set(const ConstArrayView& other) {
    set(other.data(), other.size());
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_VIEW2_INL_H_
