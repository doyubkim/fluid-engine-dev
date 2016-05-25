// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_DETAIL_SIZE3_INL_H_
#define INCLUDE_JET_DETAIL_SIZE3_INL_H_

#include <jet/math_utils.h>
#include <algorithm>
#include <cassert>

namespace jet {

// Constructors
inline Size<3>::Size() :
    x(0),
    y(0),
    z(0) {
}

inline Size<3>::Size(size_t newX, size_t newY, size_t newZ) :
    x(newX),
    y(newY),
    z(newZ) {
}

inline Size<3>::Size(const Size2& pt, size_t newZ) :
    x(pt.x),
    y(pt.y),
    z(newZ) {
}

inline Size<3>::Size(const std::initializer_list<size_t>& lst) {
    set(lst);
}

inline Size<3>::Size(const Size& v) :
    x(v.x),
    y(v.y),
    z(v.z) {
}

// Basic setters
inline void Size<3>::set(size_t newX, size_t newY, size_t newZ) {
    x = newX;
    y = newY;
    z = newZ;
}

inline void Size<3>::set(const Size2& pt, size_t newZ) {
    x = pt.x;
    y = pt.y;
    z = newZ;
}

inline void Size<3>::set(const std::initializer_list<size_t>& lst) {
    assert(lst.size() >= 3);

    auto inputElem = lst.begin();
    x = *inputElem;
    y = *(++inputElem);
    z = *(++inputElem);
}

inline void Size<3>::set(const Size& v) {
    x = v.x;
    y = v.y;
    z = v.z;
}

inline void Size<3>::setZero() {
    x = y = z = 0;
}

// Binary operators: new instance = this (+) v
inline Size<3> Size<3>::add(size_t v) const {
    return Size(x + v, y + v, z + v);
}

inline Size<3> Size<3>::add(const Size& v) const {
    return Size(x + v.x, y + v.y, z + v.z);
}

inline Size<3> Size<3>::sub(size_t v) const {
    return Size(x - v, y - v, z - v);
}

inline Size<3> Size<3>::sub(const Size& v) const {
    return Size(x - v.x, y - v.y, z - v.z);
}

inline Size<3> Size<3>::mul(size_t v) const {
    return Size(x * v, y * v, z * v);
}

inline Size<3> Size<3>::mul(const Size& v) const {
    return Size(x * v.x, y * v.y, z * v.z);
}

inline Size<3> Size<3>::div(size_t v) const {
    return Size(x / v, y / v, z / v);
}

inline Size<3> Size<3>::div(const Size& v) const {
    return Size(x / v.x, y / v.y, z / v.z);
}

// Binary operators: new instance = v (+) this
inline Size<3> Size<3>::radd(size_t v) const {
    return Size(v + x, v + y, v + z);
}

inline Size<3> Size<3>::radd(const Size& v) const {
    return Size(v.x + x, v.y + y, v.z + z);
}

inline Size<3> Size<3>::rsub(size_t v) const {
    return Size(v - x, v - y, v - z);
}

inline Size<3> Size<3>::rsub(const Size& v) const {
    return Size(v.x - x, v.y - y, v.z - z);
}

inline Size<3> Size<3>::rmul(size_t v) const {
    return Size(v * x, v * y, v * z);
}

inline Size<3> Size<3>::rmul(const Size& v) const {
    return Size(v.x * x, v.y * y, v.z * z);
}

inline Size<3> Size<3>::rdiv(size_t v) const {
    return Size(v / x, v / y, v / z);
}

inline Size<3> Size<3>::rdiv(const Size& v) const {
    return Size(v.x / x, v.y / y, v.z / z);
}

// Augmented operators: this (+)= v
inline void Size<3>::iadd(size_t v) {
    x += v;
    y += v;
    z += v;
}

inline void Size<3>::iadd(const Size& v) {
    x += v.x;
    y += v.y;
    z += v.z;
}

inline void Size<3>::isub(size_t v) {
    x -= v;
    y -= v;
    z -= v;
}

inline void Size<3>::isub(const Size& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
}

inline void Size<3>::imul(size_t v) {
    x *= v;
    y *= v;
    z *= v;
}

inline void Size<3>::imul(const Size& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
}

inline void Size<3>::idiv(size_t v) {
    x /= v;
    y /= v;
    z /= v;
}

inline void Size<3>::idiv(const Size& v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
}

// Basic getters
inline const size_t& Size<3>::at(size_t i) const {
    assert(i < 3);
    return (&x)[i];
}

inline size_t& Size<3>::at(size_t i) {
    assert(i < 3);
    return (&x)[i];
}

inline size_t Size<3>::sum() const {
    return x + y + z;
}

inline size_t Size<3>::min() const {
    return std::min(std::min(x, y), z);
}

inline size_t Size<3>::max() const {
    return std::max(std::max(x, y), z);
}

inline size_t Size<3>::dominantAxis() const {
    return (x > y) ? ((x > z) ? 0 : 2) : ((y > z) ? 1 : 2);
}

inline size_t Size<3>::subminantAxis() const {
    return (x < y) ? ((x < z) ? 0 : 2) : ((y < z) ? 1 : 2);
}

inline bool Size<3>::isEqual(const Size& other) const {
    return (x == other.x && y == other.y && z == other.z);
}

// Operators
inline size_t& Size<3>::operator[](size_t i) {
    assert(i < 3);
    return (&x)[i];
}

inline const size_t& Size<3>::operator[](size_t i) const {
    assert(i < 3);
    return (&x)[i];
}

inline Size<3>& Size<3>::operator=(const Size& v) {
    set(v);
    return (*this);
}

inline Size<3>& Size<3>::operator+=(size_t v) {
    iadd(v);
    return (*this);
}

inline Size<3>& Size<3>::operator+=(const Size& v) {
    iadd(v);
    return (*this);
}

inline Size<3>& Size<3>::operator-=(size_t v) {
    isub(v);
    return (*this);
}

inline Size<3>& Size<3>::operator-=(const Size& v) {
    isub(v);
    return (*this);
}

inline Size<3>& Size<3>::operator*=(size_t v) {
    imul(v);
    return (*this);
}

inline Size<3>& Size<3>::operator*=(const Size& v) {
    imul(v);
    return (*this);
}

inline Size<3>& Size<3>::operator/=(size_t v) {
    idiv(v);
    return (*this);
}

inline Size<3>& Size<3>::operator/=(const Size& v) {
    idiv(v);
    return (*this);
}

inline bool Size<3>::operator==(const Size& v) const {
    return isEqual(v);
}

inline bool Size<3>::operator!=(const Size& v) const {
    return !isEqual(v);
}


inline Size<3> operator+(const Size<3>& a) {
    return a;
}

inline Size<3> operator+(const Size<3>& a, size_t b) {
    return a.add(b);
}

inline Size<3> operator+(size_t a, const Size<3>& b) {
    return b.radd(a);
}

inline Size<3> operator+(const Size<3>& a, const Size<3>& b) {
    return a.add(b);
}

inline Size<3> operator-(const Size<3>& a, size_t b) {
    return a.sub(b);
}

inline Size<3> operator-(size_t a, const Size<3>& b) {
    return b.rsub(a);
}

inline Size<3> operator-(const Size<3>& a, const Size<3>& b) {
    return a.sub(b);
}

inline Size<3> operator*(const Size<3>& a, size_t b) {
    return a.mul(b);
}

inline Size<3> operator*(size_t a, const Size<3>& b) {
    return b.rmul(a);
}

inline Size<3> operator*(const Size<3>& a, const Size<3>& b) {
    return a.mul(b);
}

inline Size<3> operator/(const Size<3>& a, size_t b) {
    return a.div(b);
}

inline Size<3> operator/(size_t a, const Size<3>& b) {
    return b.rdiv(a);
}

inline Size<3> operator/(const Size<3>& a, const Size<3>& b) {
    return a.div(b);
}

inline Size<3> min(const Size<3>& a, const Size<3>& b) {
    return Size<3>(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

inline Size<3> max(const Size<3>& a, const Size<3>& b) {
    return Size<3>(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline Size<3> clamp(
    const Size<3>& v, const Size<3>& low, const Size<3>& high) {
    return Size<3>(
        clamp(v.x, low.x, high.x),
        clamp(v.y, low.y, high.y),
        clamp(v.z, low.z, high.z));
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SIZE3_INL_H_
