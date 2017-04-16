// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_POINT2_INL_H_
#define INCLUDE_JET_DETAIL_POINT2_INL_H_

#include <jet/macros.h>
#include <jet/math_utils.h>

#include <algorithm>
#include <cassert>

namespace jet {

// Constructors
template <typename T>
template <typename U>
Point<T, 2>::Point(const std::initializer_list<U>& lst) {
    set(lst);
}

// Basic setters
template <typename T>
void Point<T, 2>::set(T s) {
    x = s;
    y = s;
}

template <typename T>
void Point<T, 2>::set(T newX, T newY) {
    x = newX;
    y = newY;
}

template <typename T>
template <typename U>
void Point<T, 2>::set(const std::initializer_list<U>& lst) {
    JET_ASSERT(lst.size() >= 2);

    auto inputElem = lst.begin();
    x = static_cast<T>(*inputElem);
    y = static_cast<T>(*(++inputElem));
}

template <typename T>
void Point<T, 2>::set(const Point& v) {
    x = v.x;
    y = v.y;
}

template <typename T>
void Point<T, 2>::setZero() {
    x = y = 0;
}

// Binary operators: new instance = this (+) v
template <typename T>
Point<T, 2> Point<T, 2>::add(T v) const {
    return Point(x + v, y + v);
}

template <typename T>
Point<T, 2> Point<T, 2>::add(const Point& v) const {
    return Point(x + v.x, y + v.y);
}

template <typename T>
Point<T, 2> Point<T, 2>::sub(T v) const {
    return Point(x - v, y - v);
}

template <typename T>
Point<T, 2> Point<T, 2>::sub(const Point& v) const {
    return Point(x - v.x, y - v.y);
}

template <typename T>
Point<T, 2> Point<T, 2>::mul(T v) const {
    return Point(x * v, y * v);
}

template <typename T>
Point<T, 2> Point<T, 2>::mul(const Point& v) const {
    return Point(x * v.x, y * v.y);
}

template <typename T>
Point<T, 2> Point<T, 2>::div(T v) const {
    return Point(x / v, y / v);
}

template <typename T>
Point<T, 2> Point<T, 2>::div(const Point& v) const {
    return Point(x / v.x, y / v.y);
}

// Binary operators: new instance = v (+) this
template <typename T>
Point<T, 2> Point<T, 2>::rsub(T v) const {
    return Point(v - x, v - y);
}

template <typename T>
Point<T, 2> Point<T, 2>::rsub(const Point& v) const {
    return Point(v.x - x, v.y - y);
}

template <typename T>
Point<T, 2> Point<T, 2>::rdiv(T v) const {
    return Point(v / x, v / y);
}

template <typename T>
Point<T, 2> Point<T, 2>::rdiv(const Point& v) const {
    return Point(v.x / x, v.y / y);
}

// Augmented operators: this (+)= v
template <typename T>
void Point<T, 2>::iadd(T v) {
    x += v;
    y += v;
}

template <typename T>
void Point<T, 2>::iadd(const Point& v) {
    x += v.x;
    y += v.y;
}

template <typename T>
void Point<T, 2>::isub(T v) {
    x -= v;
    y -= v;
}

template <typename T>
void Point<T, 2>::isub(const Point& v) {
    x -= v.x;
    y -= v.y;
}

template <typename T>
void Point<T, 2>::imul(T v) {
    x *= v;
    y *= v;
}

template <typename T>
void Point<T, 2>::imul(const Point& v) {
    x *= v.x;
    y *= v.y;
}

template <typename T>
void Point<T, 2>::idiv(T v) {
    x /= v;
    y /= v;
}

template <typename T>
void Point<T, 2>::idiv(const Point& v) {
    x /= v.x;
    y /= v.y;
}

// Basic getters
template <typename T>
const T& Point<T, 2>::at(size_t i) const {
    assert(i < 2);
    return (&x)[i];
}

template <typename T>
T& Point<T, 2>::at(size_t i) {
    assert(i < 2);
    return (&x)[i];
}

template <typename T>
T Point<T, 2>::sum() const {
    return x + y;
}

template <typename T>
T Point<T, 2>::min() const {
    return std::min(x, y);
}

template <typename T>
T Point<T, 2>::max() const {
    return std::max(x, y);
}

template <typename T>
T Point<T, 2>::absmin() const {
    return jet::absmin(x, y);
}

template <typename T>
T Point<T, 2>::absmax() const {
    return jet::absmax(x, y);
}

template <typename T>
size_t Point<T, 2>::dominantAxis() const {
    return (std::fabs(x) > std::fabs(y)) ? 0 : 1;
}

template <typename T>
size_t Point<T, 2>::subminantAxis() const {
    return (std::fabs(x) < std::fabs(y)) ? 0 : 1;
}

template <typename T>
template <typename U>
Point2<U> Point<T, 2>::castTo() const {
    return Point2<U>(static_cast<U>(x), static_cast<U>(y));
}

template <typename T>
bool Point<T, 2>::isEqual(const Point& other) const {
    return (x == other.x && y == other.y);
}

// Operators
template <typename T>
T& Point<T, 2>::operator[](size_t i) {
    assert(i < 2);
    return (&x)[i];
}

template <typename T>
const T& Point<T, 2>::operator[](size_t i) const {
    assert(i < 2);
    return (&x)[i];
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator=(const std::initializer_list<T>& lst) {
    set(lst);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator=(const Point& v) {
    set(v);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator+=(T v) {
    iadd(v);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator+=(const Point& v) {
    iadd(v);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator-=(T v) {
    isub(v);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator-=(const Point& v) {
    isub(v);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator*=(T v) {
    imul(v);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator*=(const Point& v) {
    imul(v);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator/=(T v) {
    idiv(v);
    return (*this);
}

template <typename T>
Point<T, 2>& Point<T, 2>::operator/=(const Point& v) {
    idiv(v);
    return (*this);
}

template <typename T>
bool Point<T, 2>::operator==(const Point& v) const {
    return isEqual(v);
}

template <typename T>
bool Point<T, 2>::operator!=(const Point& v) const {
    return !isEqual(v);
}

// Math functions
template <typename T>
Point<T, 2> operator+(const Point<T, 2>& a) {
    return a;
}

template <typename T>
Point<T, 2> operator-(const Point<T, 2>& a) {
    return Point<T, 2>(-a.x, -a.y);
}

template <typename T>
Point<T, 2> operator+(const Point<T, 2>& a, T b) {
    return a.add(b);
}

template <typename T>
Point<T, 2> operator+(T a, const Point<T, 2>& b) {
    return b.radd(a);
}

template <typename T>
Point<T, 2> operator+(const Point<T, 2>& a, const Point<T, 2>& b) {
    return a.add(b);
}

template <typename T>
Point<T, 2> operator-(const Point<T, 2>& a, T b) {
    return a.sub(b);
}

template <typename T>
Point<T, 2> operator-(T a, const Point<T, 2>& b) {
    return b.rsub(a);
}

template <typename T>
Point<T, 2> operator-(const Point<T, 2>& a, const Point<T, 2>& b) {
    return a.sub(b);
}

template <typename T>
Point<T, 2> operator*(const Point<T, 2>& a, T b) {
    return a.mul(b);
}

template <typename T>
Point<T, 2> operator*(T a, const Point<T, 2>& b) {
    return b.rmul(a);
}

template <typename T>
Point<T, 2> operator*(const Point<T, 2>& a, const Point<T, 2>& b) {
    return a.mul(b);
}

template <typename T>
Point<T, 2> operator/(const Point<T, 2>& a, T b) {
    return a.div(b);
}

template <typename T>
Point<T, 2> operator/(T a, const Point<T, 2>& b) {
    return b.rdiv(a);
}

template <typename T>
Point<T, 2> operator/(const Point<T, 2>& a, const Point<T, 2>& b) {
    return a.div(b);
}

template <typename T>
Point<T, 2> min(const Point<T, 2>& a, const Point<T, 2>& b) {
    return Point<T, 2>(std::min(a.x, b.x), std::min(a.y, b.y));
}

template <typename T>
Point<T, 2> max(const Point<T, 2>& a, const Point<T, 2>& b) {
    return Point<T, 2>(std::max(a.x, b.x), std::max(a.y, b.y));
}

template <typename T>
Point<T, 2> clamp(
    const Point<T, 2>& v, const Point<T, 2>& low, const Point<T, 2>& high) {
    return Point<T, 2>(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
}

template <typename T>
Point<T, 2> ceil(const Point<T, 2>& a) {
    return Point<T, 2>(std::ceil(a.x), std::ceil(a.y));
}

template <typename T>
Point<T, 2> floor(const Point<T, 2>& a) {
    return Point<T, 2>(std::floor(a.x), std::floor(a.y));
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_POINT2_INL_H_
