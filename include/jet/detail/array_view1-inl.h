// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_VIEW1_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_VIEW1_INL_H_

#include <jet/array_view1.h>

#include <algorithm>

namespace jet {

namespace experimental {

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

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_VIEW1_INL_H_
