// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY3_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY3_INL_H_

#include <jet/macros.h>
#include <jet/parallel.h>

#include <algorithm>
#include <utility>  // just make cpplint happy..
#include <vector>

namespace jet {

template <typename T>
Array<T, 3>::Array() {}

template <typename T>
Array<T, 3>::Array(const Size3& size, const T& initVal) {
    resize(size, initVal);
}

template <typename T>
Array<T, 3>::Array(size_t width, size_t height, size_t depth,
                   const T& initVal) {
    resize(width, height, depth, initVal);
}

template <typename T>
Array<T, 3>::Array(const std::initializer_list<
                   std::initializer_list<std::initializer_list<T>>>& lst) {
    set(lst);
}

template <typename T>
Array<T, 3>::Array(const Array& other) {
    set(other);
}

template <typename T>
Array<T, 3>::Array(Array&& other) {
    (*this) = std::move(other);
}

template <typename T>
void Array<T, 3>::set(const T& value) {
    for (auto& v : _data) {
        v = value;
    }
}

template <typename T>
void Array<T, 3>::set(const Array& other) {
    _data.resize(other._data.size());
    std::copy(other._data.begin(), other._data.end(), _data.begin());
    _size = other._size;
}

template <typename T>
void Array<T, 3>::set(const std::initializer_list<
                      std::initializer_list<std::initializer_list<T>>>& lst) {
    size_t depth = lst.size();
    auto pageIter = lst.begin();
    size_t height = (depth > 0) ? pageIter->size() : 0;
    auto rowIter = pageIter->begin();
    size_t width = (height > 0) ? rowIter->size() : 0;
    resize(Size3(width, height, depth));
    for (size_t k = 0; k < depth; ++k) {
        rowIter = pageIter->begin();
        JET_ASSERT(height == pageIter->size());
        for (size_t j = 0; j < height; ++j) {
            auto colIter = rowIter->begin();
            JET_ASSERT(width == rowIter->size());
            for (size_t i = 0; i < width; ++i) {
                (*this)(i, j, k) = *colIter;
                ++colIter;
            }
            ++rowIter;
        }
        ++pageIter;
    }
}

template <typename T>
void Array<T, 3>::clear() {
    _size = Size3(0, 0, 0);
    _data.clear();
}

template <typename T>
void Array<T, 3>::resize(const Size3& size, const T& initVal) {
    Array grid;
    grid._data.resize(size.x * size.y * size.z, initVal);
    grid._size = size;
    size_t iMin = std::min(size.x, _size.x);
    size_t jMin = std::min(size.y, _size.y);
    size_t kMin = std::min(size.z, _size.z);
    for (size_t k = 0; k < kMin; ++k) {
        for (size_t j = 0; j < jMin; ++j) {
            for (size_t i = 0; i < iMin; ++i) {
                grid(i, j, k) = at(i, j, k);
            }
        }
    }

    swap(grid);
}

template <typename T>
void Array<T, 3>::resize(size_t width, size_t height, size_t depth,
                         const T& initVal) {
    resize(Size3(width, height, depth), initVal);
}

template <typename T>
T& Array<T, 3>::at(size_t i) {
    JET_ASSERT(i < _size.x * _size.y * _size.z);
    return _data[i];
}

template <typename T>
const T& Array<T, 3>::at(size_t i) const {
    JET_ASSERT(i < _size.x * _size.y * _size.z);
    return _data[i];
}

template <typename T>
T& Array<T, 3>::at(const Point3UI& pt) {
    return at(pt.x, pt.y, pt.z);
}

template <typename T>
const T& Array<T, 3>::at(const Point3UI& pt) const {
    return at(pt.x, pt.y, pt.z);
}

template <typename T>
T& Array<T, 3>::at(size_t i, size_t j, size_t k) {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
const T& Array<T, 3>::at(size_t i, size_t j, size_t k) const {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
Size3 Array<T, 3>::size() const {
    return _size;
}

template <typename T>
size_t Array<T, 3>::width() const {
    return _size.x;
}

template <typename T>
size_t Array<T, 3>::height() const {
    return _size.y;
}

template <typename T>
size_t Array<T, 3>::depth() const {
    return _size.z;
}

template <typename T>
T* Array<T, 3>::data() {
    return _data.data();
}

template <typename T>
const T* const Array<T, 3>::data() const {
    return _data.data();
}

template <typename T>
typename Array<T, 3>::ContainerType::iterator Array<T, 3>::begin() {
    return _data.begin();
}

template <typename T>
typename Array<T, 3>::ContainerType::const_iterator Array<T, 3>::begin() const {
    return _data.cbegin();
}

template <typename T>
typename Array<T, 3>::ContainerType::iterator Array<T, 3>::end() {
    return _data.end();
}

template <typename T>
typename Array<T, 3>::ContainerType::const_iterator Array<T, 3>::end() const {
    return _data.cend();
}

template <typename T>
ArrayAccessor3<T> Array<T, 3>::accessor() {
    return ArrayAccessor3<T>(size(), data());
}

template <typename T>
ConstArrayAccessor3<T> Array<T, 3>::constAccessor() const {
    return ConstArrayAccessor3<T>(size(), data());
}

template <typename T>
void Array<T, 3>::swap(Array& other) {
    std::swap(other._data, _data);
    std::swap(other._size, _size);
}

template <typename T>
template <typename Callback>
void Array<T, 3>::forEach(Callback func) const {
    constAccessor().forEach(func);
}

template <typename T>
template <typename Callback>
void Array<T, 3>::forEachIndex(Callback func) const {
    constAccessor().forEachIndex(func);
}

template <typename T>
template <typename Callback>
void Array<T, 3>::parallelForEach(Callback func) {
    accessor().parallelForEach(func);
}

template <typename T>
template <typename Callback>
void Array<T, 3>::parallelForEachIndex(Callback func) const {
    constAccessor().parallelForEachIndex(func);
}

template <typename T>
T& Array<T, 3>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& Array<T, 3>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
T& Array<T, 3>::operator()(size_t i, size_t j, size_t k) {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
const T& Array<T, 3>::operator()(size_t i, size_t j, size_t k) const {
    JET_ASSERT(i < _size.x && j < _size.y && k < _size.z);
    return _data[i + _size.x * (j + _size.y * k)];
}

template <typename T>
T& Array<T, 3>::operator()(const Point3UI& pt) {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y && pt.z < _size.z);
    return _data[pt.x + _size.x * (pt.y + _size.y * pt.z)];
}

template <typename T>
const T& Array<T, 3>::operator()(const Point3UI& pt) const {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y && pt.z < _size.z);
    return _data[pt.x + _size.x * (pt.y + _size.y * pt.z)];
}

template <typename T>
Array<T, 3>& Array<T, 3>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
Array<T, 3>& Array<T, 3>::operator=(const Array& other) {
    set(other);
    return *this;
}

template <typename T>
Array<T, 3>& Array<T, 3>::operator=(Array&& other) {
    _data = std::move(other._data);
    _size = other._size;
    other._size = Size3();
    return *this;
}

template <typename T>
Array<T, 3>& Array<T, 3>::operator=(
    const std::initializer_list<
        std::initializer_list<std::initializer_list<T>>>& lst) {
    set(lst);
    return *this;
}

template <typename T>
Array<T, 3>::operator ArrayAccessor3<T>() {
    return accessor();
}

template <typename T>
Array<T, 3>::operator ConstArrayAccessor3<T>() const {
    return constAccessor();
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY3_INL_H_
