// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_DETAIL_SIZE2_INL_H_
#define INCLUDE_JET_DETAIL_SIZE2_INL_H_

#include <jet/math_utils.h>
#include <algorithm>
#include <cassert>

namespace jet {

// Constructors
inline Size<2>::Size() :
    x(0),
    y(0) {
}

inline Size<2>::Size(size_t newX, size_t newY) :
    x(newX),
    y(newY) {
}

inline Size<2>::Size(const std::initializer_list<size_t>& lst) {
    set(lst);
}

inline Size<2>::Size(const Size& v) :
    x(v.x),
    y(v.y) {
}

// Basic setters
inline void Size<2>::set(size_t newX, size_t newY) {
    x = newX;
    y = newY;
}

inline void Size<2>::set(const std::initializer_list<size_t>& lst) {
    assert(lst.size() >= 2);

    auto inputElem = lst.begin();
    x = *inputElem;
    y = *(++inputElem);
}

inline void Size<2>::set(const Size& v) {
    x = v.x;
    y = v.y;
}

inline void Size<2>::setZero() {
    x = y = 0;
}

// Binary operators: new instance = this (+) v
inline Size<2> Size<2>::add(size_t v) const {
    return Size(x + v, y + v);
}

inline Size<2> Size<2>::add(const Size& v) const {
    return Size(x + v.x, y + v.y);
}

inline Size<2> Size<2>::sub(size_t v) const {
    return Size(x - v, y - v);
}

inline Size<2> Size<2>::sub(const Size& v) const {
    return Size(x - v.x, y - v.y);
}

inline Size<2> Size<2>::mul(size_t v) const {
    return Size(x * v, y * v);
}

inline Size<2> Size<2>::mul(const Size& v) const {
    return Size(x * v.x, y * v.y);
}

inline Size<2> Size<2>::div(size_t v) const {
    return Size(x / v, y / v);
}

inline Size<2> Size<2>::div(const Size& v) const {
    return Size(x / v.x, y / v.y);
}

// Binary operators: new instance = v (+) this
inline Size<2> Size<2>::radd(size_t v) const {
    return Size(v + x, v + y);
}

inline Size<2> Size<2>::radd(const Size& v) const {
    return Size(v.x + x, v.y + y);
}

inline Size<2> Size<2>::rsub(size_t v) const {
    return Size(v - x, v - y);
}

inline Size<2> Size<2>::rsub(const Size& v) const {
    return Size(v.x - x, v.y - y);
}

inline Size<2> Size<2>::rmul(size_t v) const {
    return Size(v * x, v * y);
}

inline Size<2> Size<2>::rmul(const Size& v) const {
    return Size(v.x * x, v.y * y);
}

inline Size<2> Size<2>::rdiv(size_t v) const {
    return Size(v / x, v / y);
}

inline Size<2> Size<2>::rdiv(const Size& v) const {
    return Size(v.x / x, v.y / y);
}

// Augmented operators: this (+)= v
inline void Size<2>::iadd(size_t v) {
    x += v;
    y += v;
}

inline void Size<2>::iadd(const Size& v) {
    x += v.x;
    y += v.y;
}

inline void Size<2>::isub(size_t v) {
    x -= v;
    y -= v;
}

inline void Size<2>::isub(const Size& v) {
    x -= v.x;
    y -= v.y;
}

inline void Size<2>::imul(size_t v) {
    x *= v;
    y *= v;
}

inline void Size<2>::imul(const Size& v) {
    x *= v.x;
    y *= v.y;
}

inline void Size<2>::idiv(size_t v) {
    x /= v;
    y /= v;
}

inline void Size<2>::idiv(const Size& v) {
    x /= v.x;
    y /= v.y;
}

// Basic getters
inline const size_t& Size<2>::at(size_t i) const {
    assert(i < 2);
    return (&x)[i];
}

inline size_t& Size<2>::at(size_t i) {
    assert(i < 2);
    return (&x)[i];
}

inline size_t Size<2>::sum() const {
    return x + y;
}

inline size_t Size<2>::min() const {
    return std::min(x, y);
}

inline size_t Size<2>::max() const {
    return std::max(x, y);
}

inline size_t Size<2>::dominantAxis() const {
    return (x > y) ? 0 : 1;
}

inline size_t Size<2>::subminantAxis() const {
    return (x < y) ? 0 : 1;
}

inline bool Size<2>::isEqual(const Size& other) const {
    return (x == other.x && y == other.y);
}

// Operators
inline size_t& Size<2>::operator[](size_t i) {
    assert(i < 2);
    return (&x)[i];
}

inline const size_t& Size<2>::operator[](size_t i) const {
    assert(i < 2);
    return (&x)[i];
}

inline Size<2>& Size<2>::operator=(const Size& v) {
    set(v);
    return (*this);
}

inline Size<2>& Size<2>::operator+=(size_t v) {
    iadd(v);
    return (*this);
}

inline Size<2>& Size<2>::operator+=(const Size& v) {
    iadd(v);
    return (*this);
}

inline Size<2>& Size<2>::operator-=(size_t v) {
    isub(v);
    return (*this);
}

inline Size<2>& Size<2>::operator-=(const Size& v) {
    isub(v);
    return (*this);
}

inline Size<2>& Size<2>::operator*=(size_t v) {
    imul(v);
    return (*this);
}

inline Size<2>& Size<2>::operator*=(const Size& v) {
    imul(v);
    return (*this);
}

inline Size<2>& Size<2>::operator/=(size_t v) {
    idiv(v);
    return (*this);
}

inline Size<2>& Size<2>::operator/=(const Size& v) {
    idiv(v);
    return (*this);
}

inline bool Size<2>::operator==(const Size& v) const {
    return isEqual(v);
}

inline bool Size<2>::operator!=(const Size& v) const {
    return !isEqual(v);
}

// Math functions
inline Size<2> operator+(const Size<2>& a) {
    return a;
}

inline Size<2> operator+(const Size<2>& a, size_t b) {
    return a.add(b);
}

inline Size<2> operator+(size_t a, const Size<2>& b) {
    return b.radd(a);
}

inline Size<2> operator+(const Size<2>& a, const Size<2>& b) {
    return a.add(b);
}

inline Size<2> operator-(const Size<2>& a, size_t b) {
    return a.sub(b);
}

inline Size<2> operator-(size_t a, const Size<2>& b) {
    return b.rsub(a);
}

inline Size<2> operator-(const Size<2>& a, const Size<2>& b) {
    return a.sub(b);
}

inline Size<2> operator*(const Size<2>& a, size_t b) {
    return a.mul(b);
}

inline Size<2> operator*(size_t a, const Size<2>& b) {
    return b.rmul(a);
}

inline Size<2> operator*(const Size<2>& a, const Size<2>& b) {
    return a.mul(b);
}

inline Size<2> operator/(const Size<2>& a, size_t b) {
    return a.div(b);
}

inline Size<2> operator/(size_t a, const Size<2>& b) {
    return b.rdiv(a);
}

inline Size<2> operator/(const Size<2>& a, const Size<2>& b) {
    return a.div(b);
}

inline Size<2> min(const Size<2>& a, const Size<2>& b) {
    return Size<2>(std::min(a.x, b.x), std::min(a.y, b.y));
}

inline Size<2> max(const Size<2>& a, const Size<2>& b) {
    return Size<2>(std::max(a.x, b.x), std::max(a.y, b.y));
}

inline Size<2> clamp(
    const Size<2>& v, const Size<2>& low, const Size<2>& high) {
    return Size<2>(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SIZE2_INL_H_
