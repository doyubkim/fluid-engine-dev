// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_VECTOR_INL_H_
#define INCLUDE_JET_DETAIL_VECTOR_INL_H_

#include <jet/macros.h>
#include <jet/math_utils.h>
#include <jet/vector.h>

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
template <typename E>
Vector<T, N>::Vector(const VectorExpression<T, E>& other) {
    set(other);
}

template <typename T, size_t N>
Vector<T, N>::Vector(const Vector& other) : _elements(other._elements) {}

template <typename T, size_t N>
void Vector<T, N>::set(const T& s) {
    _elements.fill(s);
}

template <typename T, size_t N>
template <typename U>
void Vector<T, N>::set(const std::initializer_list<U>& lst) {
    JET_ASSERT(lst.size() >= N);

    size_t i = 0;
    for (const auto& inputElem : lst) {
        _elements[i] = static_cast<T>(inputElem);
        ++i;
    }
}

template <typename T, size_t N>
template <typename E>
void Vector<T, N>::set(const VectorExpression<T, E>& other) {
    JET_ASSERT(size() == other.size());

    // Parallel evaluation of the expression
    const E& expression = other();
    forEachIndex([&](size_t i) { _elements[i] = expression[i]; });
}

template <typename T, size_t N>
void Vector<T, N>::swap(Vector& other) {
    std::swap(other._elements, _elements);
}

template <typename T, size_t N>
void Vector<T, N>::setZero() {
    set(T(0));
}

template <typename T, size_t N>
void Vector<T, N>::normalize() {
    idiv(length());
}

template <typename T, size_t N>
constexpr size_t Vector<T, N>::size() const {
    return N;
}

template <typename T, size_t N>
T* Vector<T, N>::data() {
    return _elements.data();
}

template <typename T, size_t N>
const T* const Vector<T, N>::data() const {
    return _elements.data();
}

template <typename T, size_t N>
typename Vector<T, N>::ContainerType::iterator Vector<T, N>::begin() {
    return _elements.begin();
}

template <typename T, size_t N>
typename Vector<T, N>::ContainerType::const_iterator Vector<T, N>::begin()
    const {
    return _elements.cbegin();
}

template <typename T, size_t N>
typename Vector<T, N>::ContainerType::iterator Vector<T, N>::end() {
    return _elements.end();
}

template <typename T, size_t N>
typename Vector<T, N>::ContainerType::const_iterator Vector<T, N>::end() const {
    return _elements.cend();
}

template <typename T, size_t N>
ArrayAccessor1<T> Vector<T, N>::accessor() {
    return ArrayAccessor1<T>(size(), data());
}

template <typename T, size_t N>
ConstArrayAccessor1<T> Vector<T, N>::constAccessor() const {
    return ConstArrayAccessor1<T>(size(), data());
}

template <typename T, size_t N>
T Vector<T, N>::at(size_t i) const {
    return _elements[i];
}

template <typename T, size_t N>
T& Vector<T, N>::at(size_t i) {
    return _elements[i];
}

template <typename T, size_t N>
T Vector<T, N>::sum() const {
    T ret = 0;
    for (T val : _elements) {
        ret += val;
    }
    return ret;
}

template <typename T, size_t N>
T Vector<T, N>::avg() const {
    return sum() / static_cast<T>(size());
}

template <typename T, size_t N>
T Vector<T, N>::min() const {
    T ret = _elements.front();
    for (T val : _elements) {
        ret = std::min(ret, val);
    }
    return ret;
}

template <typename T, size_t N>
T Vector<T, N>::max() const {
    T ret = _elements.front();
    for (T val : _elements) {
        ret = std::max(ret, val);
    }
    return ret;
}

template <typename T, size_t N>
T Vector<T, N>::absmin() const {
    T ret = _elements.front();
    for (T val : _elements) {
        ret = jet::absmin(ret, val);
    }
    return ret;
}

template <typename T, size_t N>
T Vector<T, N>::absmax() const {
    T ret = _elements.front();
    for (T val : _elements) {
        ret = jet::absmax(ret, val);
    }
    return ret;
}

template <typename T, size_t N>
size_t Vector<T, N>::dominantAxis() const {
    auto iter = std::max_element(begin(), end(), [](const T& a, const T& b) {
        return std::fabs(a) < std::fabs(b);
    });
    return iter - begin();
}

template <typename T, size_t N>
size_t Vector<T, N>::subminantAxis() const {
    auto iter = std::max_element(begin(), end(), [](const T& a, const T& b) {
        return std::fabs(a) > std::fabs(b);
    });
    return iter - begin();
}

template <typename T, size_t N>
VectorScalarDiv<T, Vector<T, N>> Vector<T, N>::normalized() const {
    T len = length();
    return VectorScalarDiv<T, Vector>(*this, len);
}

template <typename T, size_t N>
T Vector<T, N>::length() const {
    return std::sqrt(lengthSquared());
}

template <typename T, size_t N>
T Vector<T, N>::lengthSquared() const {
    return dot(*this);
}

template <typename T, size_t N>
template <typename E>
T Vector<T, N>::distanceTo(const E& other) const {
    return std::sqrt(distanceSquaredTo(other));
}

template <typename T, size_t N>
template <typename E>
T Vector<T, N>::distanceSquaredTo(const E& other) const {
    JET_ASSERT(size() == other.size());

    T ret = 0;
    for (size_t i = 0; i < N; ++i) {
        T diff = (_elements[i] - other[i]);
        ret += diff * diff;
    }

    return ret;
}

template <typename T, size_t N>
template <typename U>
VectorTypeCast<U, Vector<T, N>, T> Vector<T, N>::castTo() const {
    return VectorTypeCast<U, Vector<T, N>, T>(*this);
}

template <typename T, size_t N>
template <typename E>
bool Vector<T, N>::isEqual(const E& other) const {
    if (size() != other.size()) {
        return false;
    }

    for (size_t i = 0; i < size(); ++i) {
        if (at(i) != other[i]) {
            return false;
        }
    }
    return true;
}

template <typename T, size_t N>
template <typename E>
bool Vector<T, N>::isSimilar(const E& other, T epsilon) const {
    if (size() != other.size()) {
        return false;
    }

    for (size_t i = 0; i < size(); ++i) {
        if (std::fabs(at(i) - other[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

template <typename T, size_t N>
template <typename E>
VectorAdd<T, Vector<T, N>, E> Vector<T, N>::add(const E& v) const {
    return VectorAdd<T, Vector, E>(*this, v);
}

template <typename T, size_t N>
VectorScalarAdd<T, Vector<T, N>> Vector<T, N>::add(const T& s) const {
    return VectorScalarAdd<T, Vector>(*this, s);
}

template <typename T, size_t N>
template <typename E>
VectorSub<T, Vector<T, N>, E> Vector<T, N>::sub(const E& v) const {
    return VectorSub<T, Vector, E>(*this, v);
}

template <typename T, size_t N>
VectorScalarSub<T, Vector<T, N>> Vector<T, N>::sub(const T& s) const {
    return VectorScalarSub<T, Vector>(*this, s);
}

template <typename T, size_t N>
template <typename E>
VectorMul<T, Vector<T, N>, E> Vector<T, N>::mul(const E& v) const {
    return VectorMul<T, Vector, E>(*this, v);
}

template <typename T, size_t N>
VectorScalarMul<T, Vector<T, N>> Vector<T, N>::mul(const T& s) const {
    return VectorScalarMul<T, Vector>(*this, s);
}

template <typename T, size_t N>
template <typename E>
VectorDiv<T, Vector<T, N>, E> Vector<T, N>::div(const E& v) const {
    return VectorDiv<T, Vector, E>(*this, v);
}

template <typename T, size_t N>
VectorScalarDiv<T, Vector<T, N>> Vector<T, N>::div(const T& s) const {
    return VectorScalarDiv<T, Vector>(*this, s);
}

template <typename T, size_t N>
template <typename E>
T Vector<T, N>::dot(const E& v) const {
    JET_ASSERT(size() == v.size());

    T ret = 0;
    for (size_t i = 0; i < N; ++i) {
        ret += _elements[i] * v[i];
    }

    return ret;
}

template <typename T, size_t N>
VectorScalarRSub<T, Vector<T, N>> Vector<T, N>::rsub(const T& s) const {
    return VectorScalarRSub<T, Vector>(*this, s);
}

template <typename T, size_t N>
template <typename E>
VectorSub<T, Vector<T, N>, E> Vector<T, N>::rsub(const E& v) const {
    return VectorSub<T, Vector, E>(v, *this);
}

template <typename T, size_t N>
VectorScalarRDiv<T, Vector<T, N>> Vector<T, N>::rdiv(const T& s) const {
    return VectorScalarRDiv<T, Vector>(*this, s);
}

template <typename T, size_t N>
template <typename E>
VectorDiv<T, Vector<T, N>, E> Vector<T, N>::rdiv(const E& v) const {
    return VectorDiv<T, Vector, E>(v, *this);
}

template <typename T, size_t N>
void Vector<T, N>::iadd(const T& s) {
    set(add(s));
}

template <typename T, size_t N>
template <typename E>
void Vector<T, N>::iadd(const E& v) {
    set(add(v));
}

template <typename T, size_t N>
void Vector<T, N>::isub(const T& s) {
    set(sub(s));
}

template <typename T, size_t N>
template <typename E>
void Vector<T, N>::isub(const E& v) {
    set(sub(v));
}

template <typename T, size_t N>
void Vector<T, N>::imul(const T& s) {
    set(mul(s));
}

template <typename T, size_t N>
template <typename E>
void Vector<T, N>::imul(const E& v) {
    set(mul(v));
}

template <typename T, size_t N>
void Vector<T, N>::idiv(const T& s) {
    set(div(s));
}

template <typename T, size_t N>
template <typename E>
void Vector<T, N>::idiv(const E& v) {
    set(div(v));
}

template <typename T, size_t N>
template <typename Callback>
void Vector<T, N>::forEach(Callback func) const {
    constAccessor().forEach(func);
}

template <typename T, size_t N>
template <typename Callback>
void Vector<T, N>::forEachIndex(Callback func) const {
    constAccessor().forEachIndex(func);
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
template <typename U>
Vector<T, N>& Vector<T, N>::operator=(const std::initializer_list<U>& lst) {
    set(lst);
    return *this;
}

template <typename T, size_t N>
template <typename E>
Vector<T, N>& Vector<T, N>::operator=(const VectorExpression<T, E>& other) {
    set(other);
    return *this;
}

template <typename T, size_t N>
Vector<T, N>& Vector<T, N>::operator+=(const T& s) {
    iadd(s);
    return *this;
}

template <typename T, size_t N>
template <typename E>
Vector<T, N>& Vector<T, N>::operator+=(const E& v) {
    iadd(v);
    return *this;
}

template <typename T, size_t N>
Vector<T, N>& Vector<T, N>::operator-=(const T& s) {
    isub(s);
    return *this;
}

template <typename T, size_t N>
template <typename E>
Vector<T, N>& Vector<T, N>::operator-=(const E& v) {
    isub(v);
    return *this;
}

template <typename T, size_t N>
Vector<T, N>& Vector<T, N>::operator*=(const T& s) {
    imul(s);
    return *this;
}

template <typename T, size_t N>
template <typename E>
Vector<T, N>& Vector<T, N>::operator*=(const E& v) {
    imul(v);
    return *this;
}

template <typename T, size_t N>
Vector<T, N>& Vector<T, N>::operator/=(const T& s) {
    idiv(s);
    return *this;
}

template <typename T, size_t N>
template <typename E>
Vector<T, N>& Vector<T, N>::operator/=(const E& v) {
    idiv(v);
    return *this;
}

template <typename T, size_t N>
template <typename E>
bool Vector<T, N>::operator==(const E& v) const {
    return isEqual(v);
}

template <typename T, size_t N>
template <typename E>
bool Vector<T, N>::operator!=(const E& v) const {
    return !isEqual(v);
}

template <typename T, size_t N>
template <typename... Params>
void Vector<T, N>::setAt(size_t i, T v, Params... params) {
    _elements[i] = v;

    setAt(i + 1, params...);
}

template <typename T, size_t N>
void Vector<T, N>::setAt(size_t i, T v) {
    _elements[i] = v;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_VECTOR_INL_H_
