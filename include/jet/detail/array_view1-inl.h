// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_VIEW1_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_VIEW1_INL_H_

#include <jet/array_view1.h>

#include <algorithm>

namespace jet {

template <typename T>
ArrayView1<T>::ArrayView1() {}

template <typename T>
ArrayView1<T>::ArrayView1(T* data, size_t size) {
    set(data, size);
}

template <typename T>
ArrayView1<T>::ArrayView1(const Array1<T>& array) {
    set(array);
}

template <typename T>
ArrayView1<T>::ArrayView1(const std::vector<T>& vec) {
    set(vec);
}

template <typename T>
ArrayView1<T>::ArrayView1(const ArrayView1& other) {
    set(other);
}

template <typename T>
ArrayView1<T>::ArrayView1(ArrayView1&& other) {
    *this = std::move(other);
}

template <typename T>
void ArrayView1<T>::set(const T& value) {
    std::fill(_data, _data + _size, value);
}

template <typename T>
void ArrayView1<T>::set(T* data, size_t size) {
    _data = data;
    _size = size;
}

template <typename T>
void ArrayView1<T>::set(const Array1<T>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void ArrayView1<T>::set(const std::vector<T>& vec) {
    _data = const_cast<T*>(vec.data());
    _size = vec.size();
}

template <typename T>
void ArrayView1<T>::set(const ArrayView1& other) {
    _data = other._data;
    _size = other._size;
}

template <typename T>
void ArrayView1<T>::swap(ArrayView1& other) {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
}

template <typename T>
size_t ArrayView1<T>::size() const {
    return _size;
}

template <typename T>
T* ArrayView1<T>::data() {
    return _data;
}

template <typename T>
const T* ArrayView1<T>::data() const {
    return _data;
}

template <typename T>
T* ArrayView1<T>::begin() {
    return _data;
}

template <typename T>
const T* ArrayView1<T>::begin() const {
    return _data;
}

template <typename T>
T* ArrayView1<T>::end() {
    return _data + _size;
}

template <typename T>
const T* ArrayView1<T>::end() const {
    return _data + _size;
}

template <typename T>
T& ArrayView1<T>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& ArrayView1<T>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
ArrayView1<T>& ArrayView1<T>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
ArrayView1<T>& ArrayView1<T>::operator=(const Array1<T>& array) {
    set(array);
    return *this;
}

template <typename T>
ArrayView1<T>& ArrayView1<T>::operator=(const std::vector<T>& vec) {
    set(vec);
    return *this;
}

template <typename T>
ArrayView1<T>& ArrayView1<T>::operator=(const ArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
ArrayView1<T>& ArrayView1<T>::operator=(ArrayView1<T>&& view) {
    _data = view._data;
    _size = view._size;
    view._data = nullptr;
    view._size = 0;
    return *this;
}

//

template <typename T>
ConstArrayView1<T>::ConstArrayView1() {}

template <typename T>
ConstArrayView1<T>::ConstArrayView1(const T* data, size_t size) {
    set(data, size);
}

template <typename T>
ConstArrayView1<T>::ConstArrayView1(const Array1<T>& array) {
    set(array);
}

template <typename T>
ConstArrayView1<T>::ConstArrayView1(const std::vector<T>& vec) {
    set(vec);
}

template <typename T>
ConstArrayView1<T>::ConstArrayView1(const ArrayView1<T>& other) {
    set(other);
}

template <typename T>
ConstArrayView1<T>::ConstArrayView1(const ConstArrayView1& other) {
    set(other);
}

template <typename T>
ConstArrayView1<T>::ConstArrayView1(ConstArrayView1&& other) {
    *this = std::move(other);
}

template <typename T>
void ConstArrayView1<T>::set(const T& value) {
    std::fill(_data, _data + _size, value);
}

template <typename T>
void ConstArrayView1<T>::set(const T* data, size_t size) {
    _data = data;
    _size = size;
}

template <typename T>
void ConstArrayView1<T>::set(const Array1<T>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void ConstArrayView1<T>::set(const std::vector<T>& vec) {
    _data = const_cast<T*>(vec.data());
    _size = vec.size();
}

template <typename T>
void ConstArrayView1<T>::set(const ArrayView1<T>& other) {
    _data = other.data();
    _size = other.size();
}

template <typename T>
void ConstArrayView1<T>::set(const ConstArrayView1& other) {
    _data = other._data;
    _size = other._size;
}

template <typename T>
void ConstArrayView1<T>::swap(ConstArrayView1& other) {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
}

template <typename T>
size_t ConstArrayView1<T>::size() const {
    return _size;
}

template <typename T>
const T* ConstArrayView1<T>::data() const {
    return _data;
}

template <typename T>
const T* ConstArrayView1<T>::begin() const {
    return _data;
}

template <typename T>
const T* ConstArrayView1<T>::end() const {
    return _data + _size;
}

template <typename T>
const T& ConstArrayView1<T>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
ConstArrayView1<T>& ConstArrayView1<T>::operator=(const Array1<T>& array) {
    set(array);
    return *this;
}

template <typename T>
ConstArrayView1<T>& ConstArrayView1<T>::operator=(const std::vector<T>& vec) {
    set(vec);
    return *this;
}

template <typename T>
ConstArrayView1<T>& ConstArrayView1<T>::operator=(const ArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
ConstArrayView1<T>& ConstArrayView1<T>::operator=(
    const ConstArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
ConstArrayView1<T>& ConstArrayView1<T>::operator=(ConstArrayView1<T>&& view) {
    _data = view._data;
    _size = view._size;
    view._data = nullptr;
    view._size = 0;
    return *this;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_VIEW1_INL_H_
