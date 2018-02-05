// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_SIZE2_INL_H_
#define INCLUDE_JET_DETAIL_SIZE2_INL_H_

#include <jet/math_utils.h>
#include <jet/size2.h>

#include <algorithm>
#include <cassert>

namespace jet {

// Constructors
template <typename U>
inline Size2::Size2(const std::initializer_list<U>& lst) {
    set(lst);
}

// Basic setters
inline void Size2::set(size_t newX, size_t newY) {
    x = newX;
    y = newY;
}

template <typename U>
inline void Size2::set(const std::initializer_list<U>& lst) {
    assert(lst.size() >= 2);

    auto inputElem = lst.begin();
    x = static_cast<size_t>(*inputElem);
    y = static_cast<size_t>(*(++inputElem));
}

inline void Size2::set(const Size2& v) {
    x = v.x;
    y = v.y;
}

inline void Size2::setZero() { x = y = 0; }

// Binary operators: new instance = this (+) v
inline Size2 Size2::add(size_t v) const { return Size2(x + v, y + v); }

inline Size2 Size2::add(const Size2& v) const {
    return Size2(x + v.x, y + v.y);
}

inline Size2 Size2::sub(size_t v) const { return Size2(x - v, y - v); }

inline Size2 Size2::sub(const Size2& v) const {
    return Size2(x - v.x, y - v.y);
}

inline Size2 Size2::mul(size_t v) const { return Size2(x * v, y * v); }

inline Size2 Size2::mul(const Size2& v) const {
    return Size2(x * v.x, y * v.y);
}

inline Size2 Size2::div(size_t v) const { return Size2(x / v, y / v); }

inline Size2 Size2::div(const Size2& v) const {
    return Size2(x / v.x, y / v.y);
}

// Binary operators: new instance = v (+) this
inline Size2 Size2::rsub(size_t v) const { return Size2(v - x, v - y); }

inline Size2 Size2::rsub(const Size2& v) const {
    return Size2(v.x - x, v.y - y);
}

inline Size2 Size2::rdiv(size_t v) const { return Size2(v / x, v / y); }

inline Size2 Size2::rdiv(const Size2& v) const {
    return Size2(v.x / x, v.y / y);
}

// Augmented operators: this (+)= v
inline void Size2::iadd(size_t v) {
    x += v;
    y += v;
}

inline void Size2::iadd(const Size2& v) {
    x += v.x;
    y += v.y;
}

inline void Size2::isub(size_t v) {
    x -= v;
    y -= v;
}

inline void Size2::isub(const Size2& v) {
    x -= v.x;
    y -= v.y;
}

inline void Size2::imul(size_t v) {
    x *= v;
    y *= v;
}

inline void Size2::imul(const Size2& v) {
    x *= v.x;
    y *= v.y;
}

inline void Size2::idiv(size_t v) {
    x /= v;
    y /= v;
}

inline void Size2::idiv(const Size2& v) {
    x /= v.x;
    y /= v.y;
}

// Basic getters
inline const size_t& Size2::at(size_t i) const {
    assert(i < 2);
    return (&x)[i];
}

inline size_t& Size2::at(size_t i) {
    assert(i < 2);
    return (&x)[i];
}

inline size_t Size2::sum() const { return x + y; }

inline size_t Size2::min() const { return std::min(x, y); }

inline size_t Size2::max() const { return std::max(x, y); }

inline size_t Size2::dominantAxis() const { return (x > y) ? 0 : 1; }

inline size_t Size2::subminantAxis() const { return (x < y) ? 0 : 1; }

inline bool Size2::isEqual(const Size2& other) const {
    return (x == other.x && y == other.y);
}

// Operators
inline size_t& Size2::operator[](size_t i) {
    assert(i < 2);
    return (&x)[i];
}

inline const size_t& Size2::operator[](size_t i) const {
    assert(i < 2);
    return (&x)[i];
}

inline Size2& Size2::operator=(const Size2& v) {
    set(v);
    return (*this);
}

template <typename U>
inline Size2& Size2::operator=(const std::initializer_list<U>& lst) {
    set(lst);
    return (*this);
}

inline Size2& Size2::operator+=(size_t v) {
    iadd(v);
    return (*this);
}

inline Size2& Size2::operator+=(const Size2& v) {
    iadd(v);
    return (*this);
}

inline Size2& Size2::operator-=(size_t v) {
    isub(v);
    return (*this);
}

inline Size2& Size2::operator-=(const Size2& v) {
    isub(v);
    return (*this);
}

inline Size2& Size2::operator*=(size_t v) {
    imul(v);
    return (*this);
}

inline Size2& Size2::operator*=(const Size2& v) {
    imul(v);
    return (*this);
}

inline Size2& Size2::operator/=(size_t v) {
    idiv(v);
    return (*this);
}

inline Size2& Size2::operator/=(const Size2& v) {
    idiv(v);
    return (*this);
}

inline bool Size2::operator==(const Size2& v) const { return isEqual(v); }

inline bool Size2::operator!=(const Size2& v) const { return !isEqual(v); }

// Math functions
inline Size2 operator+(const Size2& a) { return a; }

inline Size2 operator+(const Size2& a, size_t b) { return a.add(b); }

inline Size2 operator+(size_t a, const Size2& b) { return b.add(a); }

inline Size2 operator+(const Size2& a, const Size2& b) { return a.add(b); }

inline Size2 operator-(const Size2& a, size_t b) { return a.sub(b); }

inline Size2 operator-(size_t a, const Size2& b) { return b.rsub(a); }

inline Size2 operator-(const Size2& a, const Size2& b) { return a.sub(b); }

inline Size2 operator*(const Size2& a, size_t b) { return a.mul(b); }

inline Size2 operator*(size_t a, const Size2& b) { return b.mul(a); }

inline Size2 operator*(const Size2& a, const Size2& b) { return a.mul(b); }

inline Size2 operator/(const Size2& a, size_t b) { return a.div(b); }

inline Size2 operator/(size_t a, const Size2& b) { return b.rdiv(a); }

inline Size2 operator/(const Size2& a, const Size2& b) { return a.div(b); }

inline Size2 min(const Size2& a, const Size2& b) {
    return Size2(std::min(a.x, b.x), std::min(a.y, b.y));
}

inline Size2 max(const Size2& a, const Size2& b) {
    return Size2(std::max(a.x, b.x), std::max(a.y, b.y));
}

inline Size2 clamp(const Size2& v, const Size2& low, const Size2& high) {
    return Size2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SIZE2_INL_H_
