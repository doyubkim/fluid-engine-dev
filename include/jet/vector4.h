// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VECTOR4_H_
#define INCLUDE_JET_VECTOR4_H_

#include <jet/vector3.h>
#include <algorithm>  // just make cpplint happy..
#include <limits>

namespace jet {

template <typename T>
class Vector<T, 4> final {
 public:
    T x;
    T y;
    T z;
    T w;

    // Constructors
    Vector();
    Vector(T x, T y, T z, T w);
    Vector(const Vector<T, 3>& pt, T w);
    template <typename U>
    Vector(const std::initializer_list<U>& lst);
    Vector(const Vector& v);

    // Basic setters
    void set(T s);
    void set(T x, T y, T z, T w);
    void set(const Vector<T, 3>& pt, T z);
    template <typename U>
    void set(const std::initializer_list<U>& lst);
    void set(const Vector& v);

    void setZero();
    void normalize();

    // Binary operations: new instance = this (+) v
    Vector add(T v) const;
    Vector add(const Vector& v) const;
    Vector sub(T v) const;
    Vector sub(const Vector& v) const;
    Vector mul(T v) const;
    Vector mul(const Vector& v) const;
    Vector div(T v) const;
    Vector div(const Vector& v) const;
    T dot(const Vector& v) const;

    // Binary operations: new instance = v (+) this
    Vector rsub(T v) const;
    Vector rsub(const Vector& v) const;
    Vector rdiv(T v) const;
    Vector rdiv(const Vector& v) const;

    // Augmented operations: this (+)= v
    void iadd(T v);
    void iadd(const Vector& v);
    void isub(T v);
    void isub(const Vector& v);
    void imul(T v);
    void imul(const Vector& v);
    void idiv(T v);
    void idiv(const Vector& v);

    // Basic getters
    const T& at(size_t i) const;
    T& at(size_t i);
    T sum() const;
    T avg() const;
    T min() const;
    T max() const;
    T absmin() const;
    T absmax() const;
    size_t dominantAxis() const;
    size_t subminantAxis() const;
    Vector normalized() const;
    T length() const;
    T lengthSquared() const;
    T distanceTo(const Vector& other) const;
    T distanceSquaredTo(const Vector& other) const;

    template <typename U>
    Vector<U, 4> castTo() const;

    bool isEqual(const Vector& other) const;

    bool isSimilar(
        const Vector& other,
        T epsilon = std::numeric_limits<T>::epsilon()) const;

    // Operators
    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    template <typename U>
    Vector& operator=(const std::initializer_list<U>& lst);
    Vector& operator=(const Vector& v);
    Vector& operator+=(T v);
    Vector& operator+=(const Vector& v);
    Vector& operator-=(T v);
    Vector& operator-=(const Vector& v);
    Vector& operator*=(T v);
    Vector& operator*=(const Vector& v);
    Vector& operator/=(T v);
    Vector& operator/=(const Vector& v);

    bool operator==(const Vector& v) const;
    bool operator!=(const Vector& v) const;
};


template <typename T> using Vector4 = Vector<T, 4>;

template <typename T>
Vector4<T> operator+(const Vector4<T>& a);

template <typename T>
Vector4<T> operator-(const Vector4<T>& a);

template <typename T>
Vector4<T> operator+(T a, const Vector4<T>& b);

template <typename T>
Vector4<T> operator+(const Vector4<T>& a, const Vector4<T>& b);

template <typename T>
Vector4<T> operator-(const Vector4<T>& a, T b);

template <typename T>
Vector4<T> operator-(T a, const Vector4<T>& b);

template <typename T>
Vector4<T> operator-(const Vector4<T>& a, const Vector4<T>& b);

template <typename T>
Vector4<T> operator*(const Vector4<T>& a, T b);

template <typename T>
Vector4<T> operator*(T a, const Vector4<T>& b);

template <typename T>
Vector4<T> operator*(const Vector4<T>& a, const Vector4<T>& b);

template <typename T>
Vector4<T> operator/(const Vector4<T>& a, T b);

template <typename T>
Vector4<T> operator/(T a, const Vector4<T>& b);

template <typename T>
Vector4<T> operator/(const Vector4<T>& a, const Vector4<T>& b);

template <typename T>
Vector4<T> min(const Vector4<T>& a, const Vector4<T>& b);

template <typename T>
Vector4<T> max(const Vector4<T>& a, const Vector4<T>& b);

template <typename T>
Vector4<T> clamp(
    const Vector4<T>& v, const Vector4<T>& low, const Vector4<T>& high);

template <typename T>
Vector4<T> ceil(const Vector4<T>& a);

template <typename T>
Vector4<T> floor(const Vector4<T>& a);

typedef Vector4<float> Vector4F;
typedef Vector4<double> Vector4D;

// Extensions
template <>
inline Vector4F zero<Vector4F>() {
    return Vector4F(0.f, 0.f, 0.f, 0.f);
}

template <>
inline Vector4D zero<Vector4D>() {
    return Vector4D(0.0, 0.0, 0.0, 0.0);
}

template <typename T>
struct ScalarType<Vector4<T>> {
    typedef T value;
};

template <typename T>
Vector4<T> monotonicCatmullRom(
    const Vector4<T>& v0,
    const Vector4<T>& v1,
    const Vector4<T>& v2,
    const Vector4<T>& v3,
    T f);

}  // namespace jet

#include "detail/vector4-inl.h"

#endif  // INCLUDE_JET_VECTOR4_H_
