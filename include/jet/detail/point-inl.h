// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_POINT_INL_H_
#define INCLUDE_JET_DETAIL_POINT_INL_H_

#include <jet/macros.h>
#include <cassert>

namespace jet {

template <typename T, size_t N>
Point<T, N>::Point() {
    for (auto& elem : _elements) {
        elem = static_cast<T>(0);
    }
}

template <typename T, size_t N>
template <typename... Params>
Point<T, N>::Point(Params... params) {
    static_assert(sizeof...(params) == N, "Invalid number of parameters.");

    setAt(0, params...);
}

template <typename T, size_t N>
template <typename U>
Point<T, N>::Point(const std::initializer_list<U>& lst) {
    JET_ASSERT(lst.size() >= N);

    size_t i = 0;
    for (const auto& inputElem : lst) {
        _elements[i] = static_cast<T>(inputElem);
        ++i;
    }
}

template <typename T, size_t N>
Point<T, N>::Point(const Point& other) :
    _elements(other._elements) {
}

template <typename T, size_t N>
template <typename U>
void Point<T, N>::set(const std::initializer_list<U>& lst) {
    assert(lst.size() >= N);

    size_t i = 0;
    for (const auto& inputElem : lst) {
        _elements[i] = static_cast<T>(inputElem);
        ++i;
    }
}

template <typename T, size_t N>
void Point<T, N>::set(const Point& other) {
    _elements = other._elements;
}

template <typename T, size_t N>
template <typename U>
Point<T, N>& Point<T, N>::operator=(const std::initializer_list<U>& lst) {
    set(lst);
    return *this;
}

template <typename T, size_t N>
Point<T, N>& Point<T, N>::operator=(const Point& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
const T& Point<T, N>::operator[](size_t i) const {
    return _elements[i];
}

template <typename T, size_t N>
T& Point<T, N>::operator[](size_t i) {
    return _elements[i];
}

template <typename T, size_t N>
template <typename... Params>
void Point<T, N>::setAt(size_t i, T v, Params... params) {
    _elements[i] = v;

    setAt(i+1, params...);
}

template <typename T, size_t N>
void Point<T, N>::setAt(size_t i, T v) {
    _elements[i] = v;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_POINT_INL_H_
