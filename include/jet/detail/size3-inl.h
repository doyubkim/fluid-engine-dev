// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_SIZE3_INL_H_
#define INCLUDE_JET_DETAIL_SIZE3_INL_H_

#include <jet/math_utils.h>
#include <algorithm>
#include <cassert>

namespace jet {

// Constructors
template <typename U>
inline Size3::Size3(const std::initializer_list<U>& lst) {
    set(lst);
}

// Basic setters
inline void Size3::set(size_t newX, size_t newY, size_t newZ) {
    x = newX;
    y = newY;
    z = newZ;
}

inline void Size3::set(const Size2& sz, size_t newZ) {
    x = sz.x;
    y = sz.y;
    z = newZ;
}

template <typename U>
inline void Size3::set(const std::initializer_list<U>& lst) {
    assert(lst.size() >= 3);

    auto inputElem = lst.begin();
    x = static_cast<U>(*inputElem);
    y = static_cast<U>(*(++inputElem));
    z = static_cast<U>(*(++inputElem));
}

inline void Size3::set(const Size3& v) {
    x = v.x;
    y = v.y;
    z = v.z;
}

inline void Size3::setZero() { x = y = z = 0; }

// Binary operators: new instance = this (+) v
inline Size3 Size3::add(size_t v) const { return Size3(x + v, y + v, z + v); }

inline Size3 Size3::add(const Size3& v) const {
    return Size3(x + v.x, y + v.y, z + v.z);
}

inline Size3 Size3::sub(size_t v) const { return Size3(x - v, y - v, z - v); }

inline Size3 Size3::sub(const Size3& v) const {
    return Size3(x - v.x, y - v.y, z - v.z);
}

inline Size3 Size3::mul(size_t v) const { return Size3(x * v, y * v, z * v); }

inline Size3 Size3::mul(const Size3& v) const {
    return Size3(x * v.x, y * v.y, z * v.z);
}

inline Size3 Size3::div(size_t v) const { return Size3(x / v, y / v, z / v); }

inline Size3 Size3::div(const Size3& v) const {
    return Size3(x / v.x, y / v.y, z / v.z);
}

// Binary operators: new instance = v (+) this
inline Size3 Size3::rsub(size_t v) const { return Size3(v - x, v - y, v - z); }

inline Size3 Size3::rsub(const Size3& v) const {
    return Size3(v.x - x, v.y - y, v.z - z);
}

inline Size3 Size3::rdiv(size_t v) const { return Size3(v / x, v / y, v / z); }

inline Size3 Size3::rdiv(const Size3& v) const {
    return Size3(v.x / x, v.y / y, v.z / z);
}

// Augmented operators: this (+)= v
inline void Size3::iadd(size_t v) {
    x += v;
    y += v;
    z += v;
}

inline void Size3::iadd(const Size3& v) {
    x += v.x;
    y += v.y;
    z += v.z;
}

inline void Size3::isub(size_t v) {
    x -= v;
    y -= v;
    z -= v;
}

inline void Size3::isub(const Size3& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
}

inline void Size3::imul(size_t v) {
    x *= v;
    y *= v;
    z *= v;
}

inline void Size3::imul(const Size3& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
}

inline void Size3::idiv(size_t v) {
    x /= v;
    y /= v;
    z /= v;
}

inline void Size3::idiv(const Size3& v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
}

// Basic getters
inline const size_t& Size3::at(size_t i) const {
    assert(i < 3);
    return (&x)[i];
}

inline size_t& Size3::at(size_t i) {
    assert(i < 3);
    return (&x)[i];
}

inline size_t Size3::sum() const { return x + y + z; }

inline size_t Size3::min() const { return std::min(std::min(x, y), z); }

inline size_t Size3::max() const { return std::max(std::max(x, y), z); }

inline size_t Size3::dominantAxis() const {
    return (x > y) ? ((x > z) ? 0 : 2) : ((y > z) ? 1 : 2);
}

inline size_t Size3::subminantAxis() const {
    return (x < y) ? ((x < z) ? 0 : 2) : ((y < z) ? 1 : 2);
}

inline bool Size3::isEqual(const Size3& other) const {
    return (x == other.x && y == other.y && z == other.z);
}

// Operators
inline size_t& Size3::operator[](size_t i) {
    assert(i < 3);
    return (&x)[i];
}

inline const size_t& Size3::operator[](size_t i) const {
    assert(i < 3);
    return (&x)[i];
}

inline Size3& Size3::operator=(const Size3& v) {
    set(v);
    return (*this);
}

inline Size3& Size3::operator+=(size_t v) {
    iadd(v);
    return (*this);
}

inline Size3& Size3::operator+=(const Size3& v) {
    iadd(v);
    return (*this);
}

inline Size3& Size3::operator-=(size_t v) {
    isub(v);
    return (*this);
}

inline Size3& Size3::operator-=(const Size3& v) {
    isub(v);
    return (*this);
}

inline Size3& Size3::operator*=(size_t v) {
    imul(v);
    return (*this);
}

inline Size3& Size3::operator*=(const Size3& v) {
    imul(v);
    return (*this);
}

inline Size3& Size3::operator/=(size_t v) {
    idiv(v);
    return (*this);
}

inline Size3& Size3::operator/=(const Size3& v) {
    idiv(v);
    return (*this);
}

inline bool Size3::operator==(const Size3& v) const { return isEqual(v); }

inline bool Size3::operator!=(const Size3& v) const { return !isEqual(v); }

inline Size3 operator+(const Size3& a) { return a; }

inline Size3 operator+(const Size3& a, size_t b) { return a.add(b); }

inline Size3 operator+(size_t a, const Size3& b) { return b.add(a); }

inline Size3 operator+(const Size3& a, const Size3& b) { return a.add(b); }

inline Size3 operator-(const Size3& a, size_t b) { return a.sub(b); }

inline Size3 operator-(size_t a, const Size3& b) { return b.rsub(a); }

inline Size3 operator-(const Size3& a, const Size3& b) { return a.sub(b); }

inline Size3 operator*(const Size3& a, size_t b) { return a.mul(b); }

inline Size3 operator*(size_t a, const Size3& b) { return b.mul(a); }

inline Size3 operator*(const Size3& a, const Size3& b) { return a.mul(b); }

inline Size3 operator/(const Size3& a, size_t b) { return a.div(b); }

inline Size3 operator/(size_t a, const Size3& b) { return b.rdiv(a); }

inline Size3 operator/(const Size3& a, const Size3& b) { return a.div(b); }

inline Size3 min(const Size3& a, const Size3& b) {
    return Size3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

inline Size3 max(const Size3& a, const Size3& b) {
    return Size3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline Size3 clamp(const Size3& v, const Size3& low, const Size3& high) {
    return Size3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y),
                 clamp(v.z, low.z, high.z));
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SIZE3_INL_H_
