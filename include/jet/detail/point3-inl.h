// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_POINT3_INL_H_
#define INCLUDE_JET_DETAIL_POINT3_INL_H_

#include <jet/macros.h>
#include <jet/math_utils.h>

#include <algorithm>
#include <cassert>

namespace jet {

// Constructors
template <typename T>
template <typename U>
Point<T, 3>::Point(const std::initializer_list<U>& lst) {
    set(lst);
}

// Basic setters
template <typename T>
void Point<T, 3>::set(T s) {
    x = s;
    y = s;
    z = s;
}

template <typename T>
void Point<T, 3>::set(T newX, T newY, T newZ) {
    x = newX;
    y = newY;
    z = newZ;
}

template <typename T>
void Point<T, 3>::set(const Point2<T>& pt, T newZ) {
    x = pt.x;
    y = pt.y;
    z = newZ;
}

template <typename T>
template <typename U>
void Point<T, 3>::set(const std::initializer_list<U>& lst) {
    assert(lst.size() >= 3);

    auto inputElem = lst.begin();
    x = static_cast<T>(*inputElem);
    y = static_cast<T>(*(++inputElem));
    z = static_cast<T>(*(++inputElem));
}

template <typename T>
void Point<T, 3>::set(const Point& v) {
    x = v.x;
    y = v.y;
    z = v.z;
}

template <typename T>
void Point<T, 3>::setZero() {
    x = y = z = 0;
}

// Binary operators: new instance = this (+) v
template <typename T>
Point<T, 3> Point<T, 3>::add(T v) const {
    return Point(x + v, y + v, z + v);
}

template <typename T>
Point<T, 3> Point<T, 3>::add(const Point& v) const {
    return Point(x + v.x, y + v.y, z + v.z);
}

template <typename T>
Point<T, 3> Point<T, 3>::sub(T v) const {
    return Point(x - v, y - v, z - v);
}

template <typename T>
Point<T, 3> Point<T, 3>::sub(const Point& v) const {
    return Point(x - v.x, y - v.y, z - v.z);
}

template <typename T>
Point<T, 3> Point<T, 3>::mul(T v) const {
    return Point(x * v, y * v, z * v);
}

template <typename T>
Point<T, 3> Point<T, 3>::mul(const Point& v) const {
    return Point(x * v.x, y * v.y, z * v.z);
}

template <typename T>
Point<T, 3> Point<T, 3>::div(T v) const {
    return Point(x / v, y / v, z / v);
}

template <typename T>
Point<T, 3> Point<T, 3>::div(const Point& v) const {
    return Point(x / v.x, y / v.y, z / v.z);
}

// Binary operators: new instance = v (+) this
template <typename T>
Point<T, 3> Point<T, 3>::rsub(T v) const {
    return Point(v - x, v - y, v - z);
}

template <typename T>
Point<T, 3> Point<T, 3>::rsub(const Point& v) const {
    return Point(v.x - x, v.y - y, v.z - z);
}

template <typename T>
Point<T, 3> Point<T, 3>::rdiv(T v) const {
    return Point(v / x, v / y, v / z);
}

template <typename T>
Point<T, 3> Point<T, 3>::rdiv(const Point& v) const {
    return Point(v.x / x, v.y / y, v.z / z);
}

// Augmented operators: this (+)= v
template <typename T>
void Point<T, 3>::iadd(T v) {
    x += v;
    y += v;
    z += v;
}

template <typename T>
void Point<T, 3>::iadd(const Point& v) {
    x += v.x;
    y += v.y;
    z += v.z;
}

template <typename T>
void Point<T, 3>::isub(T v) {
    x -= v;
    y -= v;
    z -= v;
}

template <typename T>
void Point<T, 3>::isub(const Point& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
}

template <typename T>
void Point<T, 3>::imul(T v) {
    x *= v;
    y *= v;
    z *= v;
}

template <typename T>
void Point<T, 3>::imul(const Point& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
}

template <typename T>
void Point<T, 3>::idiv(T v) {
    x /= v;
    y /= v;
    z /= v;
}

template <typename T>
void Point<T, 3>::idiv(const Point& v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
}

// Basic getters
template <typename T>
const T& Point<T, 3>::at(size_t i) const {
    assert(i < 3);
    return (&x)[i];
}

template <typename T>
T& Point<T, 3>::at(size_t i) {
    assert(i < 3);
    return (&x)[i];
}

template <typename T>
T Point<T, 3>::sum() const {
    return x + y + z;
}

template <typename T>
T Point<T, 3>::min() const {
    return std::min(std::min(x, y), z);
}

template <typename T>
T Point<T, 3>::max() const {
    return std::max(std::max(x, y), z);
}

template <typename T>
T Point<T, 3>::absmin() const {
    return jet::absmin(jet::absmin(x, y), z);
}

template <typename T>
T Point<T, 3>::absmax() const {
    return jet::absmax(jet::absmax(x, y), z);
}

template <typename T>
size_t Point<T, 3>::dominantAxis() const {
    return (std::fabs(x) > std::fabs(y))
               ? ((std::fabs(x) > std::fabs(z)) ? 0 : 2)
               : ((std::fabs(y) > std::fabs(z)) ? 1 : 2);
}

template <typename T>
size_t Point<T, 3>::subminantAxis() const {
    return (std::fabs(x) < std::fabs(y))
               ? ((std::fabs(x) < std::fabs(z)) ? 0 : 2)
               : ((std::fabs(y) < std::fabs(z)) ? 1 : 2);
}

template <typename T>
template <typename U>
Point3<U> Point<T, 3>::castTo() const {
    return Point3<U>(static_cast<U>(x), static_cast<U>(y), static_cast<U>(z));
}

template <typename T>
bool Point<T, 3>::isEqual(const Point& other) const {
    return (x == other.x && y == other.y && z == other.z);
}

// Operators
template <typename T>
T& Point<T, 3>::operator[](size_t i) {
    assert(i < 3);
    return (&x)[i];
}

template <typename T>
const T& Point<T, 3>::operator[](size_t i) const {
    assert(i < 3);
    return (&x)[i];
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator=(const std::initializer_list<T>& lst) {
    set(lst);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator=(const Point& v) {
    set(v);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator+=(T v) {
    iadd(v);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator+=(const Point& v) {
    iadd(v);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator-=(T v) {
    isub(v);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator-=(const Point& v) {
    isub(v);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator*=(T v) {
    imul(v);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator*=(const Point& v) {
    imul(v);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator/=(T v) {
    idiv(v);
    return (*this);
}

template <typename T>
Point<T, 3>& Point<T, 3>::operator/=(const Point& v) {
    idiv(v);
    return (*this);
}

template <typename T>
bool Point<T, 3>::operator==(const Point& v) const {
    return isEqual(v);
}

template <typename T>
bool Point<T, 3>::operator!=(const Point& v) const {
    return !isEqual(v);
}

template <typename T>
Point<T, 3> operator+(const Point<T, 3>& a) {
    return a;
}

template <typename T>
Point<T, 3> operator-(const Point<T, 3>& a) {
    return Point<T, 3>(-a.x, -a.y, -a.z);
}

template <typename T>
Point<T, 3> operator+(const Point<T, 3>& a, T b) {
    return a.add(b);
}

template <typename T>
Point<T, 3> operator+(T a, const Point<T, 3>& b) {
    return b.radd(a);
}

template <typename T>
Point<T, 3> operator+(const Point<T, 3>& a, const Point<T, 3>& b) {
    return a.add(b);
}

template <typename T>
Point<T, 3> operator-(const Point<T, 3>& a, T b) {
    return a.sub(b);
}

template <typename T>
Point<T, 3> operator-(T a, const Point<T, 3>& b) {
    return b.rsub(a);
}

template <typename T>
Point<T, 3> operator-(const Point<T, 3>& a, const Point<T, 3>& b) {
    return a.sub(b);
}

template <typename T>
Point<T, 3> operator*(const Point<T, 3>& a, T b) {
    return a.mul(b);
}

template <typename T>
Point<T, 3> operator*(T a, const Point<T, 3>& b) {
    return b.rmul(a);
}

template <typename T>
Point<T, 3> operator*(const Point<T, 3>& a, const Point<T, 3>& b) {
    return a.mul(b);
}

template <typename T>
Point<T, 3> operator/(const Point<T, 3>& a, T b) {
    return a.div(b);
}

template <typename T>
Point<T, 3> operator/(T a, const Point<T, 3>& b) {
    return b.rdiv(a);
}

template <typename T>
Point<T, 3> operator/(const Point<T, 3>& a, const Point<T, 3>& b) {
    return a.div(b);
}

template <typename T>
Point<T, 3> min(const Point<T, 3>& a, const Point<T, 3>& b) {
    return Point<T, 3>(std::min(a.x, b.x), std::min(a.y, b.y),
                       std::min(a.z, b.z));
}

template <typename T>
Point<T, 3> max(const Point<T, 3>& a, const Point<T, 3>& b) {
    return Point<T, 3>(std::max(a.x, b.x), std::max(a.y, b.y),
                       std::max(a.z, b.z));
}

template <typename T>
Point<T, 3> clamp(const Point<T, 3>& v, const Point<T, 3>& low,
                  const Point<T, 3>& high) {
    return Point<T, 3>(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y),
                       clamp(v.z, low.z, high.z));
}

template <typename T>
Point<T, 3> ceil(const Point<T, 3>& a) {
    return Point<T, 3>(std::ceil(a.x), std::ceil(a.y), std::ceil(a.z));
}

template <typename T>
Point<T, 3> floor(const Point<T, 3>& a) {
    return Point<T, 3>(std::floor(a.x), std::floor(a.y), std::floor(a.z));
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_POINT3_INL_H_
