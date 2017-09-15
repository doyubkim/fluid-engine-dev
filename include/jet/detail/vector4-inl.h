// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_VECTOR4_INL_H_
#define INCLUDE_JET_DETAIL_VECTOR4_INL_H_

#include <jet/macros.h>
#include <jet/math_utils.h>
#include <algorithm>
#include <limits>
#include <tuple>

namespace jet {

// Constructors
template <typename T>
template <typename U>
Vector<T, 4>::Vector(const std::initializer_list<U>& lst) {
    set(lst);
}

// Basic setters
template <typename T>
void Vector<T, 4>::set(T s) {
    x = s;
    y = s;
    z = s;
    w = s;
}

template <typename T>
void Vector<T, 4>::set(T newX, T newY, T newZ, T newW) {
    x = newX;
    y = newY;
    z = newZ;
    w = newW;
}

template <typename T>
void Vector<T, 4>::set(const Vector<T, 3>& pt, T newW) {
    x = pt.x;
    y = pt.y;
    z = pt.z;
    w = newW;
}

template <typename T>
template <typename U>
void Vector<T, 4>::set(const std::initializer_list<U>& lst) {
    JET_ASSERT(lst.size() >= 4);

    auto inputElem = lst.begin();
    x = static_cast<T>(*inputElem);
    y = static_cast<T>(*(++inputElem));
    z = static_cast<T>(*(++inputElem));
    w = static_cast<T>(*(++inputElem));
}

template <typename T>
void Vector<T, 4>::set(const Vector& v) {
    x = v.x;
    y = v.y;
    z = v.z;
    w = v.w;
}

template <typename T>
void Vector<T, 4>::setZero() {
    x = y = z = w = 0;
}

template <typename T>
void Vector<T, 4>::normalize() {
    T l = length();
    x /= l;
    y /= l;
    z /= l;
    w /= l;
}

// Binary operators: new instance = this (+) v
template <typename T>
Vector<T, 4> Vector<T, 4>::add(T v) const {
    return Vector(x + v, y + v, z + v, w + v);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::add(const Vector& v) const {
    return Vector(x + v.x, y + v.y, z + v.z, w + v.w);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::sub(T v) const {
    return Vector(x - v, y - v, z - v, w - v);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::sub(const Vector& v) const {
    return Vector(x - v.x, y - v.y, z - v.z, w - v.w);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::mul(T v) const {
    return Vector(x * v, y * v, z * v, w * v);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::mul(const Vector& v) const {
    return Vector(x * v.x, y * v.y, z * v.z, w * v.w);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::div(T v) const {
    return Vector(x / v, y / v, z / v, w / v);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::div(const Vector& v) const {
    return Vector(x / v.x, y / v.y, z / v.z, w / v.w);
}

template <typename T>
T Vector<T, 4>::dot(const Vector& v) const {
    return x * v.x + y * v.y + z * v.z + w * v.w;
}

// Binary operators: new instance = v (+) this
template <typename T>
Vector<T, 4> Vector<T, 4>::rsub(T v) const {
    return Vector(v - x, v - y, v - z, v - w);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::rsub(const Vector& v) const {
    return Vector(v.x - x, v.y - y, v.z - z, v.w - w);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::rdiv(T v) const {
    return Vector(v / x, v / y, v / z, v / w);
}

template <typename T>
Vector<T, 4> Vector<T, 4>::rdiv(const Vector& v) const {
    return Vector(v.x / x, v.y / y, v.z / z, v.w / w);
}

// Augmented operators: this (+)= v
template <typename T>
void Vector<T, 4>::iadd(T v) {
    x += v;
    y += v;
    z += v;
    w += v;
}

template <typename T>
void Vector<T, 4>::iadd(const Vector& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
}

template <typename T>
void Vector<T, 4>::isub(T v) {
    x -= v;
    y -= v;
    z -= v;
    w -= v;
}

template <typename T>
void Vector<T, 4>::isub(const Vector& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
}

template <typename T>
void Vector<T, 4>::imul(T v) {
    x *= v;
    y *= v;
    z *= v;
    w *= v;
}

template <typename T>
void Vector<T, 4>::imul(const Vector& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;
}

template <typename T>
void Vector<T, 4>::idiv(T v) {
    x /= v;
    y /= v;
    z /= v;
    w /= v;
}

template <typename T>
void Vector<T, 4>::idiv(const Vector& v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;
}

// Basic getters
template <typename T>
const T& Vector<T, 4>::at(size_t i) const {
    JET_ASSERT(i < 4);
    return (&x)[i];
}

template <typename T>
T& Vector<T, 4>::at(size_t i) {
    JET_ASSERT(i < 4);
    return (&x)[i];
}

template <typename T>
T Vector<T, 4>::sum() const {
    return x + y + z + w;
}

template <typename T>
T Vector<T, 4>::avg() const {
    return (x + y + z + w) / 4;
}

template <typename T>
T Vector<T, 4>::min() const {
    return std::min(std::min(x, y), std::min(z, w));
}

template <typename T>
T Vector<T, 4>::max() const {
    return std::max(std::max(x, y), std::max(z, w));
}

template <typename T>
T Vector<T, 4>::absmin() const {
    return jet::absmin(jet::absmin(x, y), jet::absmin(z, w));
}

template <typename T>
T Vector<T, 4>::absmax() const {
    return jet::absmax(jet::absmax(x, y), jet::absmax(z, w));
}

template <typename T>
size_t Vector<T, 4>::dominantAxis() const {
    return (std::fabs(x) > std::fabs(y))
               ? ((std::fabs(x) > std::fabs(z))
                      ? ((std::fabs(x) > std::fabs(w)) ? 0 : 3)
                      : ((std::fabs(x) > std::fabs(w)) ? 2 : 3))
               : ((std::fabs(y) > std::fabs(z))
                      ? ((std::fabs(y) > std::fabs(w)) ? 1 : 3)
                      : ((std::fabs(z) > std::fabs(w)) ? 2 : 3));
}

template <typename T>
size_t Vector<T, 4>::subminantAxis() const {
    return (std::fabs(x) < std::fabs(y))
               ? ((std::fabs(x) < std::fabs(z))
                      ? ((std::fabs(x) < std::fabs(w)) ? 0 : 3)
                      : ((std::fabs(x) < std::fabs(w)) ? 2 : 3))
               : ((std::fabs(y) < std::fabs(z))
                      ? ((std::fabs(y) < std::fabs(w)) ? 1 : 3)
                      : ((std::fabs(z) < std::fabs(w)) ? 2 : 3));
}

template <typename T>
Vector<T, 4> Vector<T, 4>::normalized() const {
    T l = length();
    return Vector(x / l, y / l, z / l, w / l);
}

template <typename T>
T Vector<T, 4>::length() const {
    return std::sqrt(x * x + y * y + z * z + w * w);
}

template <typename T>
T Vector<T, 4>::lengthSquared() const {
    return x * x + y * y + z * z + w * w;
}

template <typename T>
T Vector<T, 4>::distanceTo(const Vector<T, 4>& other) const {
    return sub(other).length();
}

template <typename T>
T Vector<T, 4>::distanceSquaredTo(const Vector<T, 4>& other) const {
    return sub(other).lengthSquared();
}

template <typename T>
template <typename U>
Vector<U, 4> Vector<T, 4>::castTo() const {
    return Vector<U, 4>(static_cast<U>(x), static_cast<U>(y), static_cast<U>(z),
                        static_cast<U>(w));
}

template <typename T>
bool Vector<T, 4>::isEqual(const Vector& other) const {
    return x == other.x && y == other.y && z == other.z && w == other.w;
}

template <typename T>
bool Vector<T, 4>::isSimilar(const Vector& other, T epsilon) const {
    return (std::fabs(x - other.x) < epsilon) &&
           (std::fabs(y - other.y) < epsilon) &&
           (std::fabs(z - other.z) < epsilon) &&
           (std::fabs(w - other.w) < epsilon);
}

// Operators
template <typename T>
T& Vector<T, 4>::operator[](size_t i) {
    JET_ASSERT(i < 4);
    return (&x)[i];
}

template <typename T>
const T& Vector<T, 4>::operator[](size_t i) const {
    JET_ASSERT(i < 4);
    return (&x)[i];
}

template <typename T>
template <typename U>
Vector<T, 4>& Vector<T, 4>::operator=(const std::initializer_list<U>& lst) {
    set(lst);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator=(const Vector& v) {
    set(v);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator+=(T v) {
    iadd(v);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator+=(const Vector& v) {
    iadd(v);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator-=(T v) {
    isub(v);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator-=(const Vector& v) {
    isub(v);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator*=(T v) {
    imul(v);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator*=(const Vector& v) {
    imul(v);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator/=(T v) {
    idiv(v);
    return (*this);
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator/=(const Vector& v) {
    idiv(v);
    return (*this);
}

template <typename T>
bool Vector<T, 4>::operator==(const Vector& v) const {
    return x == v.x && y == v.y && z == v.z && w == v.w;
}

template <typename T>
bool Vector<T, 4>::operator!=(const Vector& v) const {
    return x != v.x || y != v.y || z != v.z || w != v.w;
}

template <typename T>
Vector<T, 4> operator+(const Vector<T, 4>& a) {
    return a;
}

template <typename T>
Vector<T, 4> operator-(const Vector<T, 4>& a) {
    return Vector<T, 4>(-a.x, -a.y, -a.z, -a.w);
}

template <typename T>
Vector<T, 4> operator+(const Vector<T, 4>& a, T b) {
    return a.add(b);
}

template <typename T>
Vector<T, 4> operator+(T a, const Vector<T, 4>& b) {
    return b.add(a);
}

template <typename T>
Vector<T, 4> operator+(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    return a.add(b);
}

template <typename T>
Vector<T, 4> operator-(const Vector<T, 4>& a, T b) {
    return a.sub(b);
}

template <typename T>
Vector<T, 4> operator-(T a, const Vector<T, 4>& b) {
    return b.rsub(a);
}

template <typename T>
Vector<T, 4> operator-(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    return a.sub(b);
}

template <typename T>
Vector<T, 4> operator*(const Vector<T, 4>& a, T b) {
    return a.mul(b);
}

template <typename T>
Vector<T, 4> operator*(T a, const Vector<T, 4>& b) {
    return b.mul(a);
}

template <typename T>
Vector<T, 4> operator*(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    return a.mul(b);
}

template <typename T>
Vector<T, 4> operator/(const Vector<T, 4>& a, T b) {
    return a.div(b);
}

template <typename T>
Vector<T, 4> operator/(T a, const Vector<T, 4>& b) {
    return b.rdiv(a);
}

template <typename T>
Vector<T, 4> operator/(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    return a.div(b);
}

template <typename T>
Vector<T, 4> min(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    return Vector<T, 4>(std::min(a.x, b.x), std::min(a.y, b.y),
                        std::min(a.z, b.z), std::min(a.w, b.w));
}

template <typename T>
Vector<T, 4> max(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    return Vector<T, 4>(std::max(a.x, b.x), std::max(a.y, b.y),
                        std::max(a.z, b.z), std::max(a.w, b.w));
}

template <typename T>
Vector<T, 4> clamp(const Vector<T, 4>& v, const Vector<T, 4>& low,
                   const Vector<T, 4>& high) {
    return Vector<T, 4>(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y),
                        clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
}

template <typename T>
Vector<T, 4> ceil(const Vector<T, 4>& a) {
    return Vector<T, 4>(std::ceil(a.x), std::ceil(a.y), std::ceil(a.z),
                        std::ceil(a.w));
}

template <typename T>
Vector<T, 4> floor(const Vector<T, 4>& a) {
    return Vector<T, 4>(std::floor(a.x), std::floor(a.y), std::floor(a.z),
                        std::floor(a.w));
}

// Extensions
template <typename T>
Vector<T, 4> monotonicCatmullRom(const Vector<T, 4>& v0, const Vector<T, 4>& v1,
                                 const Vector<T, 4>& v2, const Vector<T, 4>& v3,
                                 T f) {
    static const T two = static_cast<T>(2);
    static const T three = static_cast<T>(3);

    Vector<T, 4> d1 = (v2 - v0) / two;
    Vector<T, 4> d2 = (v3 - v1) / two;
    Vector<T, 4> D1 = v2 - v1;

    if (std::fabs(D1.x) < std::numeric_limits<float>::epsilon() ||
        sign(D1.x) != sign(d1.x) || sign(D1.x) != sign(d2.x)) {
        d1.x = d2.x = 0;
    }

    if (std::fabs(D1.y) < std::numeric_limits<float>::epsilon() ||
        sign(D1.y) != sign(d1.y) || sign(D1.y) != sign(d2.y)) {
        d1.y = d2.y = 0;
    }

    if (std::fabs(D1.z) < std::numeric_limits<float>::epsilon() ||
        sign(D1.z) != sign(d1.z) || sign(D1.z) != sign(d2.z)) {
        d1.z = d2.z = 0;
    }

    if (std::fabs(D1.w) < std::numeric_limits<float>::epsilon() ||
        sign(D1.w) != sign(d1.w) || sign(D1.w) != sign(d2.w)) {
        d1.w = d2.w = 0;
    }

    Vector<T, 4> a3 = d1 + d2 - two * D1;
    Vector<T, 4> a2 = three * D1 - two * d1 - d2;
    Vector<T, 4> a1 = d1;
    Vector<T, 4> a0 = v1;

    return a3 * cubic(f) + a2 * square(f) + a1 * f + a0;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_VECTOR4_INL_H_
