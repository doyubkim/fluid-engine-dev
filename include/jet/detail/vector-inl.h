// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_DETAIL_VECTOR_INL_H_
#define INCLUDE_JET_DETAIL_VECTOR_INL_H_

#include <cassert>

namespace jet {

template <typename T, size_t N>
Vector<T, N>::Vector() {
    for (auto& elem : _elements) {
        elem = static_cast<T>(0);
    }
}

template <typename T, size_t N>
template <typename... Params>
Vector<T, N>::Vector(Params... params) {
    static_assert(sizeof...(params) == N, "Invalid number of parameters.");

    setAt(0, params...);
}

template <typename T, size_t N>
template <typename U>
Vector<T, N>::Vector(const std::initializer_list<U>& lst) {
    set(lst);
}

template <typename T, size_t N>
Vector<T, N>::Vector(const Vector& other) :
    _elements(other._elements) {
}

template <typename T, size_t N>
template <typename U>
void Vector<T, N>::set(const std::initializer_list<U>& lst) {
    assert(lst.size() >= N);

    size_t i = 0;
    for (const auto& inputElem : lst) {
        _elements[i] = static_cast<T>(inputElem);
        ++i;
    }
}

template <typename T, size_t N>
void Vector<T, N>::set(const Vector& other) {
    _elements = other._elements;
}

template <typename T, size_t N>
template <typename U>
Vector<T, N>& Vector<T, N>::operator=(const std::initializer_list<U>& lst) {
    set(lst);
    return *this;
}

template <typename T, size_t N>
Vector<T, N>& Vector<T, N>::operator=(const Vector& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
const T& Vector<T, N>::operator[](size_t i) const {
    return _elements[i];
}

template <typename T, size_t N>
T& Vector<T, N>::operator[](size_t i) {
    return _elements[i];
}

template <typename T, size_t N>
template <typename... Params>
void Vector<T, N>::setAt(size_t i, T v, Params... params) {
    _elements[i] = v;

    setAt(i+1, params...);
}

template <typename T, size_t N>
void Vector<T, N>::setAt(size_t i, T v) {
    _elements[i] = v;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_VECTOR_INL_H_
