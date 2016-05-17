// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VECTOR3_H_
#define INCLUDE_JET_VECTOR3_H_

#include <jet/vector2.h>
#include <limits>
#include <tuple>

namespace jet {

template <typename T>
class Vector<T, 3> final {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "Vector only can be instantiated with floating point types");

    T x;
    T y;
    T z;

    // Constructors
    Vector();
    explicit Vector(T x, T y, T z);
    explicit Vector(const Vector2<T>& pt, T z);
    template <typename U>
    Vector(const std::initializer_list<U>& lst);
    Vector(const Vector& v);

    // Basic setters
    void set(T s);
    void set(T x, T y, T z);
    void set(const Vector2<T>& pt, T z);
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
    Vector cross(const Vector& v) const;

    // Binary operations: new instance = v (+) this
    Vector rsub(T v) const;
    Vector rsub(const Vector& v) const;
    Vector rdiv(T v) const;
    Vector rdiv(const Vector& v) const;
    Vector rcross(const Vector& v) const;

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

    //! Returns the reflection vector to the surface with given surface normal.
    Vector reflected(const Vector& normal) const;

    //! Returns the projected vector to the surface with given surface normal.
    Vector projected(const Vector& normal) const;

    //! Returns the tangential vector for this vector.
    std::tuple<Vector, Vector> tangential() const;

    template <typename U>
    Vector<U, 3> castTo() const;

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


template <typename T> using Vector3 = Vector<T, 3>;

template <typename T>
Vector3<T> operator+(const Vector3<T>& a);

template <typename T>
Vector3<T> operator-(const Vector3<T>& a);

template <typename T>
Vector3<T> operator+(T a, const Vector3<T>& b);

template <typename T>
Vector3<T> operator+(const Vector3<T>& a, const Vector3<T>& b);

template <typename T>
Vector3<T> operator-(const Vector3<T>& a, T b);

template <typename T>
Vector3<T> operator-(T a, const Vector3<T>& b);

template <typename T>
Vector3<T> operator-(const Vector3<T>& a, const Vector3<T>& b);

template <typename T>
Vector3<T> operator*(const Vector3<T>& a, T b);

template <typename T>
Vector3<T> operator*(T a, const Vector3<T>& b);

template <typename T>
Vector3<T> operator*(const Vector3<T>& a, const Vector3<T>& b);

template <typename T>
Vector3<T> operator/(const Vector3<T>& a, T b);

template <typename T>
Vector3<T> operator/(T a, const Vector3<T>& b);

template <typename T>
Vector3<T> operator/(const Vector3<T>& a, const Vector3<T>& b);

template <typename T>
Vector3<T> min(const Vector3<T>& a, const Vector3<T>& b);

template <typename T>
Vector3<T> max(const Vector3<T>& a, const Vector3<T>& b);

template <typename T>
Vector3<T> clamp(
    const Vector3<T>& v, const Vector3<T>& low, const Vector3<T>& high);

template <typename T>
Vector3<T> ceil(const Vector3<T>& a);

template <typename T>
Vector3<T> floor(const Vector3<T>& a);

typedef Vector3<float> Vector3F;
typedef Vector3<double> Vector3D;

// Extensions
template <>
inline Vector3F zero<Vector3F>() {
    return Vector3F(0.f, 0.f, 0.f);
}

template <>
inline Vector3D zero<Vector3D>() {
    return Vector3D(0.0, 0.0, 0.0);
}

template <typename T>
struct ScalarType<Vector3<T>> {
    typedef T value;
};

template <typename T>
Vector3<T> monotonicCatmullRom(
    const Vector3<T>& v0,
    const Vector3<T>& v1,
    const Vector3<T>& v2,
    const Vector3<T>& v3,
    T f);

}  // namespace jet

#include "detail/vector3-inl.h"

#endif  // INCLUDE_JET_VECTOR3_H_

