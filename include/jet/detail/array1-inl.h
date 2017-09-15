// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY1_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY1_INL_H_

#include <jet/constants.h>
#include <jet/parallel.h>

#include <algorithm>
#include <utility>  // just make cpplint happy..
#include <vector>

namespace jet {

template <typename T>
Array<T, 1>::Array() {}

template <typename T>
Array<T, 1>::Array(size_t size, const T& initVal) {
    resize(size, initVal);
}

template <typename T>
Array<T, 1>::Array(const std::initializer_list<T>& lst) {
    set(lst);
}

template <typename T>
Array<T, 1>::Array(const Array& other) {
    set(other);
}

template <typename T>
Array<T, 1>::Array(Array&& other) {
    (*this) = std::move(other);
}

template <typename T>
void Array<T, 1>::set(const T& value) {
    for (auto& v : _data) {
        v = value;
    }
}

template <typename T>
void Array<T, 1>::set(const Array& other) {
    _data.resize(other._data.size());
    std::copy(other._data.begin(), other._data.end(), _data.begin());
}

template <typename T>
void Array<T, 1>::set(const std::initializer_list<T>& lst) {
    size_t size = lst.size();
    resize(size);
    auto colIter = lst.begin();
    for (size_t i = 0; i < size; ++i) {
        (*this)[i] = *colIter;
        ++colIter;
    }
}

template <typename T>
void Array<T, 1>::clear() {
    _data.clear();
}

template <typename T>
void Array<T, 1>::resize(size_t size, const T& initVal) {
    _data.resize(size, initVal);
}

template <typename T>
T& Array<T, 1>::at(size_t i) {
    assert(i < size());
    return _data[i];
}

template <typename T>
const T& Array<T, 1>::at(size_t i) const {
    assert(i < size());
    return _data[i];
}

template <typename T>
size_t Array<T, 1>::size() const {
    return _data.size();
}

template <typename T>
T* Array<T, 1>::data() {
    return _data.data();
}

template <typename T>
const T* const Array<T, 1>::data() const {
    return _data.data();
}

template <typename T>
typename Array<T, 1>::ContainerType::iterator Array<T, 1>::begin() {
    return _data.begin();
}

template <typename T>
typename Array<T, 1>::ContainerType::const_iterator Array<T, 1>::begin() const {
    return _data.cbegin();
}

template <typename T>
typename Array<T, 1>::ContainerType::iterator Array<T, 1>::end() {
    return _data.end();
}

template <typename T>
typename Array<T, 1>::ContainerType::const_iterator Array<T, 1>::end() const {
    return _data.cend();
}

template <typename T>
ArrayAccessor1<T> Array<T, 1>::accessor() {
    return ArrayAccessor1<T>(size(), data());
}

template <typename T>
ConstArrayAccessor1<T> Array<T, 1>::constAccessor() const {
    return ConstArrayAccessor1<T>(size(), data());
}

template <typename T>
void Array<T, 1>::swap(Array& other) {
    std::swap(other._data, _data);
}

template <typename T>
void Array<T, 1>::append(const T& newVal) {
    _data.push_back(newVal);
}

template <typename T>
void Array<T, 1>::append(const Array& other) {
    _data.insert(_data.end(), other._data.begin(), other._data.end());
}

template <typename T>
template <typename Callback>
void Array<T, 1>::forEach(Callback func) const {
    constAccessor().forEach(func);
}

template <typename T>
template <typename Callback>
void Array<T, 1>::forEachIndex(Callback func) const {
    constAccessor().forEachIndex(func);
}

template <typename T>
template <typename Callback>
void Array<T, 1>::parallelForEach(Callback func) {
    accessor().parallelForEach(func);
}

template <typename T>
template <typename Callback>
void Array<T, 1>::parallelForEachIndex(Callback func) const {
    constAccessor().parallelForEachIndex(func);
}

template <typename T>
T& Array<T, 1>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& Array<T, 1>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
Array<T, 1>& Array<T, 1>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
Array<T, 1>& Array<T, 1>::operator=(const Array& other) {
    set(other);
    return *this;
}

template <typename T>
Array<T, 1>& Array<T, 1>::operator=(Array&& other) {
    _data = std::move(other._data);
    return *this;
}

template <typename T>
Array<T, 1>& Array<T, 1>::operator=(const std::initializer_list<T>& lst) {
    set(lst);
    return *this;
}

template <typename T>
Array<T, 1>::operator ArrayAccessor1<T>() {
    return accessor();
}

template <typename T>
Array<T, 1>::operator ConstArrayAccessor1<T>() const {
    return constAccessor();
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY1_INL_H_
