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
ArrayView<T, 1>::ArrayView() {}

template <typename T>
ArrayView<T, 1>::ArrayView(T* data, size_t size) {
    set(data, size);
}

template <typename T>
ArrayView<T, 1>::ArrayView(const Array1<T>& array) {
    set(array);
}

template <typename T>
ArrayView<T, 1>::ArrayView(const std::vector<T>& vec) {
    set(vec);
}

template <typename T>
ArrayView<T, 1>::ArrayView(const ArrayView& other) {
    set(other);
}

template <typename T>
ArrayView<T, 1>::ArrayView(ArrayView&& other) {
    *this = std::move(other);
}

template <typename T>
void ArrayView<T, 1>::set(const T& value) {
    std::fill(_data, _data + _size, value);
}

template <typename T>
void ArrayView<T, 1>::set(T* data, size_t size) {
    _data = data;
    _size = size;
}

template <typename T>
void ArrayView<T, 1>::set(const Array1<T>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void ArrayView<T, 1>::set(const std::vector<T>& vec) {
    _data = const_cast<T*>(vec.data());
    _size = vec.size();
}

template <typename T>
void ArrayView<T, 1>::set(const ArrayView& other) {
    _data = other._data;
    _size = other._size;
}

template <typename T>
void ArrayView<T, 1>::swap(ArrayView& other) {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
}

template <typename T>
size_t ArrayView<T, 1>::size() const {
    return _size;
}

template <typename T>
T* ArrayView<T, 1>::data() {
    return _data;
}

template <typename T>
const T* ArrayView<T, 1>::data() const {
    return _data;
}

template <typename T>
T* ArrayView<T, 1>::begin() {
    return _data;
}

template <typename T>
const T* ArrayView<T, 1>::begin() const {
    return _data;
}

template <typename T>
T* ArrayView<T, 1>::end() {
    return _data + _size;
}

template <typename T>
const T* ArrayView<T, 1>::end() const {
    return _data + _size;
}

template <typename T>
T& ArrayView<T, 1>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& ArrayView<T, 1>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
ArrayView1<T>& ArrayView<T, 1>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
ArrayView1<T>& ArrayView<T, 1>::operator=(const Array1<T>& array) {
    set(array);
    return *this;
}

template <typename T>
ArrayView1<T>& ArrayView<T, 1>::operator=(const std::vector<T>& vec) {
    set(vec);
    return *this;
}

template <typename T>
ArrayView1<T>& ArrayView<T, 1>::operator=(const ArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
ArrayView1<T>& ArrayView<T, 1>::operator=(ArrayView1<T>&& view) {
    _data = view._data;
    _size = view._size;
    view._data = nullptr;
    view._size = 0;
    return *this;
}

//

template <typename T>
ConstArrayView<T, 1>::ConstArrayView() {}

template <typename T>
ConstArrayView<T, 1>::ConstArrayView(const T* data, size_t size) {
    set(data, size);
}

template <typename T>
ConstArrayView<T, 1>::ConstArrayView(const Array1<T>& array) {
    set(array);
}

template <typename T>
ConstArrayView<T, 1>::ConstArrayView(const std::vector<T>& vec) {
    set(vec);
}

template <typename T>
ConstArrayView<T, 1>::ConstArrayView(const ArrayView1<T>& other) {
    set(other);
}

template <typename T>
ConstArrayView<T, 1>::ConstArrayView(const ConstArrayView& other) {
    set(other);
}

template <typename T>
ConstArrayView<T, 1>::ConstArrayView(ConstArrayView&& other) {
    *this = std::move(other);
}

template <typename T>
void ConstArrayView<T, 1>::set(const T& value) {
    std::fill(_data, _data + _size, value);
}

template <typename T>
void ConstArrayView<T, 1>::set(const T* data, size_t size) {
    _data = data;
    _size = size;
}

template <typename T>
void ConstArrayView<T, 1>::set(const Array1<T>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void ConstArrayView<T, 1>::set(const std::vector<T>& vec) {
    _data = const_cast<T*>(vec.data());
    _size = vec.size();
}

template <typename T>
void ConstArrayView<T, 1>::set(const ArrayView1<T>& other) {
    _data = other.data();
    _size = other.size();
}

template <typename T>
void ConstArrayView<T, 1>::set(const ConstArrayView& other) {
    _data = other._data;
    _size = other._size;
}

template <typename T>
void ConstArrayView<T, 1>::swap(ConstArrayView& other) {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
}

template <typename T>
size_t ConstArrayView<T, 1>::size() const {
    return _size;
}

template <typename T>
const T* ConstArrayView<T, 1>::data() const {
    return _data;
}

template <typename T>
const T* ConstArrayView<T, 1>::begin() const {
    return _data;
}

template <typename T>
const T* ConstArrayView<T, 1>::end() const {
    return _data + _size;
}

template <typename T>
const T& ConstArrayView<T, 1>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
ConstArrayView<T, 1>& ConstArrayView<T, 1>::operator=(const Array1<T>& array) {
    set(array);
    return *this;
}

template <typename T>
ConstArrayView<T, 1>& ConstArrayView<T, 1>::operator=(const std::vector<T>& vec) {
    set(vec);
    return *this;
}

template <typename T>
ConstArrayView<T, 1>& ConstArrayView<T, 1>::operator=(const ArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
ConstArrayView<T, 1>& ConstArrayView<T, 1>::operator=(
    const ConstArrayView& view) {
    set(view);
    return *this;
}

template <typename T>
ConstArrayView<T, 1>& ConstArrayView<T, 1>::operator=(ConstArrayView&& view) {
    _data = view._data;
    _size = view._size;
    view._data = nullptr;
    view._size = 0;
    return *this;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_VIEW1_INL_H_
