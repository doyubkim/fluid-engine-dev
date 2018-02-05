// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_ACCESSOR2_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_ACCESSOR2_INL_H_

#include <jet/macros.h>
#include <jet/parallel.h>
#include <utility>  // just make cpplint happy..

namespace jet {

template <typename T>
ArrayAccessor<T, 2>::ArrayAccessor() : _data(nullptr) {
}

template <typename T>
ArrayAccessor<T, 2>::ArrayAccessor(const Size2& size, T* const data) {
    reset(size, data);
}

template <typename T>
ArrayAccessor<T, 2>::ArrayAccessor(size_t width, size_t height, T* const data) {
    reset(width, height, data);
}

template <typename T>
ArrayAccessor<T, 2>::ArrayAccessor(const ArrayAccessor& other) {
    set(other);
}

template <typename T>
void ArrayAccessor<T, 2>::set(const ArrayAccessor& other) {
    reset(other._size, other._data);
}

template <typename T>
void ArrayAccessor<T, 2>::reset(const Size2& size, T* const data) {
    _size = size;
    _data = data;
}

template <typename T>
void ArrayAccessor<T, 2>::reset(size_t width, size_t height, T* const data) {
    reset(Size2(width, height), data);
}

template <typename T>
T& ArrayAccessor<T, 2>::at(size_t i) {
    JET_ASSERT(i < _size.x*_size.y);
    return _data[i];
}

template <typename T>
const T& ArrayAccessor<T, 2>::at(size_t i) const {
    JET_ASSERT(i < _size.x*_size.y);
    return _data[i];
}

template <typename T>
T& ArrayAccessor<T, 2>::at(const Point2UI& pt) {
    return at(pt.x, pt.y);
}

template <typename T>
const T& ArrayAccessor<T, 2>::at(const Point2UI& pt) const {
    return at(pt.x, pt.y);
}

template <typename T>
T& ArrayAccessor<T, 2>::at(size_t i, size_t j) {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
const T& ArrayAccessor<T, 2>::at(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
T* const ArrayAccessor<T, 2>::begin() const {
    return _data;
}

template <typename T>
T* const ArrayAccessor<T, 2>::end() const {
    return _data + _size.x * _size.y;
}

template <typename T>
T* ArrayAccessor<T, 2>::begin() {
    return _data;
}

template <typename T>
T* ArrayAccessor<T, 2>::end() {
    return _data + _size.x * _size.y;
}

template <typename T>
Size2 ArrayAccessor<T, 2>::size() const {
    return _size;
}

template <typename T>
size_t ArrayAccessor<T, 2>::width() const {
    return _size.x;
}

template <typename T>
size_t ArrayAccessor<T, 2>::height() const {
    return _size.y;
}

template <typename T>
T* const ArrayAccessor<T, 2>::data() const {
    return _data;
}

template <typename T>
void ArrayAccessor<T, 2>::swap(ArrayAccessor& other) {
    std::swap(other._data, _data);
    std::swap(other._size, _size);
}

template <typename T>
template <typename Callback>
void ArrayAccessor<T, 2>::forEach(Callback func) const {
    for (size_t j = 0; j < _size.y; ++j) {
        for (size_t i = 0; i < _size.x; ++i) {
            func(at(i, j));
        }
    }
}

template <typename T>
template <typename Callback>
void ArrayAccessor<T, 2>::forEachIndex(Callback func) const {
    for (size_t j = 0; j < _size.y; ++j) {
        for (size_t i = 0; i < _size.x; ++i) {
            func(i, j);
        }
    }
}

template <typename T>
template <typename Callback>
void ArrayAccessor<T, 2>::parallelForEach(Callback func) {
    parallelFor(kZeroSize, _size.x, kZeroSize, _size.y,
        [&](size_t i, size_t j) {
            func(at(i, j));
        });
}

template <typename T>
template <typename Callback>
void ArrayAccessor<T, 2>::parallelForEachIndex(Callback func) const {
    parallelFor(kZeroSize, _size.x, kZeroSize, _size.y, func);
}

template <typename T>
size_t ArrayAccessor<T, 2>::index(const Point2UI& pt) const {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y);
    return pt.x + _size.x * pt.y;
}

template <typename T>
size_t ArrayAccessor<T, 2>::index(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return i + _size.x * j;
}

template <typename T>
T& ArrayAccessor<T, 2>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& ArrayAccessor<T, 2>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
T& ArrayAccessor<T, 2>::operator()(const Point2UI &pt) {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y);
    return _data[pt.x + _size.x * pt.y];
}

template <typename T>
const T& ArrayAccessor<T, 2>::operator()(const Point2UI &pt) const {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y);
    return _data[pt.x + _size.x * pt.y];
}

template <typename T>
T& ArrayAccessor<T, 2>::operator()(size_t i, size_t j) {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
const T& ArrayAccessor<T, 2>::operator()(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
ArrayAccessor<T, 2>& ArrayAccessor<T, 2>::operator=(
    const ArrayAccessor& other) {
    set(other);
    return *this;
}

template <typename T>
ArrayAccessor<T, 2>::operator ConstArrayAccessor<T, 2>() const {
    return ConstArrayAccessor<T, 2>(*this);
}


template <typename T>
ConstArrayAccessor<T, 2>::ConstArrayAccessor() : _data(nullptr) {
}

template <typename T>
ConstArrayAccessor<T, 2>::ConstArrayAccessor(
    const Size2& size, const T* const data) {
    _size = size;
    _data = data;
}

template <typename T>
ConstArrayAccessor<T, 2>::ConstArrayAccessor(
    size_t width, size_t height, const T* const data) {
    _size = Size2(width, height);
    _data = data;
}

template <typename T>
ConstArrayAccessor<T, 2>::ConstArrayAccessor(const ArrayAccessor<T, 2>& other) {
    _size = other.size();
    _data = other.data();
}

template <typename T>
ConstArrayAccessor<T, 2>::ConstArrayAccessor(const ConstArrayAccessor& other) {
    _size = other._size;
    _data = other._data;
}

template <typename T>
const T& ConstArrayAccessor<T, 2>::at(size_t i) const {
    JET_ASSERT(i < _size.x*_size.y);
    return _data[i];
}

template <typename T>
const T& ConstArrayAccessor<T, 2>::at(const Point2UI& pt) const {
    return at(pt.x, pt.y);
}

template <typename T>
const T& ConstArrayAccessor<T, 2>::at(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

template <typename T>
const T* const ConstArrayAccessor<T, 2>::begin() const {
    return _data;
}

template <typename T>
const T* const ConstArrayAccessor<T, 2>::end() const {
    return _data + _size.x * _size.y;
}

template <typename T>
Size2 ConstArrayAccessor<T, 2>::size() const {
    return _size;
}

template <typename T>
size_t ConstArrayAccessor<T, 2>::width() const {
    return _size.x;
}

template <typename T>
size_t ConstArrayAccessor<T, 2>::height() const {
    return _size.y;
}

template <typename T>
const T* const ConstArrayAccessor<T, 2>::data() const {
    return _data;
}

template <typename T>
template <typename Callback>
void ConstArrayAccessor<T, 2>::forEach(Callback func) const {
    for (size_t j = 0; j < _size.y; ++j) {
        for (size_t i = 0; i < _size.x; ++i) {
            func(at(i, j));
        }
    }
}

template <typename T>
template <typename Callback>
void ConstArrayAccessor<T, 2>::forEachIndex(Callback func) const {
    for (size_t j = 0; j < _size.y; ++j) {
        for (size_t i = 0; i < _size.x; ++i) {
            func(i, j);
        }
    }
}

template <typename T>
template <typename Callback>
void ConstArrayAccessor<T, 2>::parallelForEachIndex(Callback func) const {
    parallelFor(kZeroSize, _size.x, kZeroSize, _size.y, func);
}

template <typename T>
size_t ConstArrayAccessor<T, 2>::index(const Point2UI& pt) const {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y);
    return pt.x + _size.x * pt.y;
}

template <typename T>
size_t ConstArrayAccessor<T, 2>::index(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return i + _size.x * j;
}

template <typename T>
const T& ConstArrayAccessor<T, 2>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
const T& ConstArrayAccessor<T, 2>::operator()(const Point2UI &pt) const {
    JET_ASSERT(pt.x < _size.x && pt.y < _size.y);
    return _data[pt.x + _size.x * pt.y];
}

template <typename T>
const T& ConstArrayAccessor<T, 2>::operator()(size_t i, size_t j) const {
    JET_ASSERT(i < _size.x && j < _size.y);
    return _data[i + _size.x * j];
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_ACCESSOR2_INL_H_
