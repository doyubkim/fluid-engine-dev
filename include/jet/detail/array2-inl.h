// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY2_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY2_INL_H_

#include <jet/array2.h>
#include <jet/array_view2.h>
#include <jet/macros.h>
#include <jet/parallel.h>

#include <algorithm>
#include <vector>

namespace jet {

template <typename T>
Array<T, 2>::Array() {}

template <typename T>
Array<T, 2>::Array(const Size2& size, const T& initVal) {
    resize(size, initVal);
}

template <typename T>
Array<T, 2>::Array(size_t width, size_t height, const T& initVal) {
    resize(width, height, initVal);
}

template <typename T>
Array<T, 2>::Array(const ConstArrayView<T, 2>& view) {
    set(view);
}

template <typename T>
Array<T, 2>::Array(const std::initializer_list<std::initializer_list<T>>& lst) {
    set(lst);
}

template <typename T>
Array<T, 2>::Array(const Array& other) {
    set(other);
}

template <typename T>
Array<T, 2>::Array(Array&& other) {
    (*this) = std::move(other);
}

template <typename T>
void Array<T, 2>::set(const T& value) {
    for (auto& v : _data) {
        v = value;
    }
}

template <typename T>
void Array<T, 2>::set(const Array& other) {
    _data.resize(other._data.size());
    std::copy(other._data.begin(), other._data.end(), _data.begin());
    _size = other._size;
}

template <typename T>
void Array<T, 2>::set(const ConstArrayView<T, 2>& view) {
    Size2 sz = view.size();
    Array<T, 2> temp(sz);
    size_t n = sz.x * sz.y;
    for (size_t i = 0; i < n; ++i) {
        temp[i] = view[i];
    }
    (*this) = std::move(temp);
}

template <typename T>
void Array<T, 2>::set(
    const std::initializer_list<std::initializer_list<T>>& lst) {
    size_t height = lst.size();
    size_t width = (height > 0) ? lst.begin()->size() : 0;
    resize(Size2(width, height));
    auto rowIter = lst.begin();
    for (size_t j = 0; j < height; ++j) {
        JET_ASSERT(width == rowIter->size());
        auto colIter = rowIter->begin();
        for (size_t i = 0; i < width; ++i) {
            (*this)(i, j) = *colIter;
            ++colIter;
        }
        ++rowIter;
    }
}

template <typename T>
void Array<T, 2>::clear() {
    _data.clear();
    _size = Size2(0, 0);
}

template <typename T>
void Array<T, 2>::resize(const Size2& size, const T& initVal) {
    Array grid;
    grid._data.resize(size.x * size.y, initVal);
    grid._size = size;
    size_t iMin = std::min(size.x, _size.x);
    size_t jMin = std::min(size.y, _size.y);
    for (size_t j = 0; j < jMin; ++j) {
        for (size_t i = 0; i < iMin; ++i) {
            grid(i, j) = at(i, j);
        }
    }

    swap(grid);
}

template <typename T>
void Array<T, 2>::resize(size_t width, size_t height, const T& initVal) {
    resize(Size2(width, height), initVal);
}

template <typename T>
T& Array<T, 2>::at(size_t i) {
    JET_ASSERT(i < _size.x * _size.y);
    return _data[i];
}

template <typename T>
const T& Array<T, 2>::at(size_t i) const {
    JET_ASSERT(i < _size.x * _size.y);
    return _data[i];
}

template <typename T>
T& Array<T, 2>::at(const Size2& pt) {
    return at(pt.x, pt.y);
}

template <typename T>
const T& Array<T, 2>::at(const Size2& pt) const {
    return at(pt.x, pt.y);
}

template <typename T>
T& Array<T, 2>::at(size_t i, size_t j) {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
const T& Array<T, 2>::at(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
Size2 Array<T, 2>::size() const {
    return _size;
}

template <typename T>
size_t Array<T, 2>::width() const {
    return _size.x;
}

template <typename T>
size_t Array<T, 2>::height() const {
    return _size.y;
}

template <typename T>
T* Array<T, 2>::data() {
    return _data.data();
}

template <typename T>
const T* Array<T, 2>::data() const {
    return _data.data();
}

template <typename T>
typename Array<T, 2>::ContainerType::iterator Array<T, 2>::begin() {
    return _data.begin();
}

template <typename T>
typename Array<T, 2>::ContainerType::const_iterator Array<T, 2>::begin() const {
    return _data.cbegin();
}

template <typename T>
typename Array<T, 2>::ContainerType::iterator Array<T, 2>::end() {
    return _data.end();
}

template <typename T>
typename Array<T, 2>::ContainerType::const_iterator Array<T, 2>::end() const {
    return _data.cend();
}

template <typename T>
ArrayAccessor2<T> Array<T, 2>::accessor() {
    return ArrayAccessor2<T>(size(), data());
}

template <typename T>
ConstArrayAccessor2<T> Array<T, 2>::constAccessor() const {
    return ConstArrayAccessor2<T>(size(), data());
}

template <typename T>
ArrayView<T, 2> Array<T, 2>::view() {
    return ArrayView<T, 2>(*this);
}

template <typename T>
ConstArrayView<T, 2> Array<T, 2>::view() const {
    return ConstArrayView<T, 2>(*this);
}

template <typename T>
void Array<T, 2>::swap(Array& other) {
    std::swap(other._data, _data);
    std::swap(other._size, _size);
}

template <typename T>
template <typename Callback>
void Array<T, 2>::forEach(Callback func) const {
    constAccessor().forEach(func);
}

template <typename T>
template <typename Callback>
void Array<T, 2>::forEachIndex(Callback func) const {
    constAccessor().forEachIndex(func);
}

template <typename T>
template <typename Callback>
void Array<T, 2>::parallelForEach(Callback func) {
    accessor().parallelForEach(func);
}

template <typename T>
template <typename Callback>
void Array<T, 2>::parallelForEachIndex(Callback func) const {
    constAccessor().parallelForEachIndex(func);
}

template <typename T>
T& Array<T, 2>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& Array<T, 2>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
T& Array<T, 2>::operator()(size_t i, size_t j) {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
const T& Array<T, 2>::operator()(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
T& Array<T, 2>::operator()(const Size2& pt) {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y);
    return _data[pt.x + _size.x * pt.y];
}

template <typename T>
const T& Array<T, 2>::operator()(const Size2& pt) const {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y);
    return _data[pt.x + _size.x * pt.y];
}

template <typename T>
Array<T, 2>& Array<T, 2>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
Array<T, 2>& Array<T, 2>::operator=(const Array& other) {
    set(other);
    return *this;
}

template <typename T>
Array<T, 2>& Array<T, 2>::operator=(const ConstArrayView<T, 2>& other) {
    set(other);
    return *this;
}

template <typename T>
Array<T, 2>& Array<T, 2>::operator=(Array&& other) {
    _data = std::move(other._data);
    _size = other._size;
    other._size = Size2();
    return *this;
}

template <typename T>
Array<T, 2>& Array<T, 2>::operator=(
    const std::initializer_list<std::initializer_list<T>>& lst) {
    set(lst);
    return *this;
}

template <typename T>
Array<T, 2>::operator ArrayAccessor2<T>() {
    return accessor();
}

template <typename T>
Array<T, 2>::operator ConstArrayAccessor2<T>() const {
    return constAccessor();
}

template <typename T>
Array<T, 2>::operator ArrayView<T, 2>() {
    return view();
}

template <typename T>
Array<T, 2>::operator ConstArrayView<T, 2>() const {
    return view();
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY2_INL_H_
