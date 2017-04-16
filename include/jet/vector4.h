// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VECTOR4_H_
#define INCLUDE_JET_VECTOR4_H_

#include <jet/vector3.h>

#include <algorithm>  // just make cpplint happy..
#include <limits>

namespace jet {

//!
//! \brief 4-D vector class.
//!
//! This class defines simple 4-D vector data.
//!
//! \tparam T - Type of the element
//!
template <typename T>
class Vector<T, 4> final {
 public:
    static_assert(std::is_floating_point<T>::value,
                  "Vector only can be instantiated with floating point types");

    //! X (or the first) component of the vector.
    T x;

    //! Y (or the second) component of the vector.
    T y;

    //! Z (or the third) component of the vector.
    T z;

    //! W (or the fourth) component of the vector.
    T w;

    // MARK: Constructors

    //! Constructs default vector (0, 0, 0, 0).
    constexpr Vector() : x(0), y(0), z(0), w(0) {}

    //! Constructs vector with given parameters \p x_, \p y_, \p z_, and \p w_.
    constexpr Vector(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}

    //! Constructs vector with a 3-D vector (x, y, and z) and a scalar (w).
    constexpr Vector(const Vector3<T>& v, T w_)
        : x(v.x), y(v.y), z(v.z), w(w_) {}

    //! Constructs vector with initializer list.
    template <typename U>
    Vector(const std::initializer_list<U>& lst);

    //! Copy constructor.
    constexpr Vector(const Vector& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    // MARK: Basic setters

    //! Set both x, y, z, and w components to \p s.
    void set(T s);

    //! Set x, y, z, and w components with given parameters.
    void set(T x, T y, T z, T w);

    //! Set x, y, z, and w components with \p pt.x, \p pt.y, \p z, and \p w.
    void set(const Vector<T, 3>& pt, T z);

    //! Set x, y, z, and w components with given initializer list.
    template <typename U>
    void set(const std::initializer_list<U>& lst);

    //! Set x, y, z, and w with other vector \p v.
    void set(const Vector& v);

    //! Set all x, y, z, and w to zero.
    void setZero();

    //! Normalizes this vector.
    void normalize();

    // MARK: Binary operations: new instance = this (+) v

    //! Computes this + (v, v, v).
    Vector add(T v) const;

    //! Computes this + (v.x, v.y, v.z).
    Vector add(const Vector& v) const;

    //! Computes this - (v, v, v).
    Vector sub(T v) const;

    //! Computes this - (v.x, v.y, v.z).
    Vector sub(const Vector& v) const;

    //! Computes this * (v, v, v).
    Vector mul(T v) const;

    //! Computes this * (v.x, v.y, v.z).
    Vector mul(const Vector& v) const;
    //! Computes this / (v, v, v).
    Vector div(T v) const;

    //! Computes this / (v.x, v.y, v.z).
    Vector div(const Vector& v) const;

    //! Computes dot product.
    T dot(const Vector& v) const;

    // MARK: Binary operations: new instance = v (+) this

    //! Computes (v, v, v) - this.
    Vector rsub(T v) const;

    //! Computes (v.x, v.y, v.z) - this.
    Vector rsub(const Vector& v) const;

    //! Computes (v, v, v) / this.
    Vector rdiv(T v) const;

    //! Computes (v.x, v.y, v.z) / this.
    Vector rdiv(const Vector& v) const;

    // MARK: Augmented operations: this (+)= v

    //! Computes this += (v, v, v).
    void iadd(T v);

    //! Computes this += (v.x, v.y, v.z).
    void iadd(const Vector& v);

    //! Computes this -= (v, v, v).
    void isub(T v);

    //! Computes this -= (v.x, v.y, v.z).
    void isub(const Vector& v);

    //! Computes this *= (v, v, v).
    void imul(T v);

    //! Computes this *= (v.x, v.y, v.z).
    void imul(const Vector& v);

    //! Computes this /= (v, v, v).
    void idiv(T v);

    //! Computes this /= (v.x, v.y, v.z).
    void idiv(const Vector& v);

    // MARK: Basic getters

    //! Returns const reference to the \p i -th element of the vector.
    const T& at(size_t i) const;

    //! Returns reference to the \p i -th element of the vector.
    T& at(size_t i);

    //! Returns the sum of all the components (i.e. x + y + z + w).
    T sum() const;

    //! Returns the average of all the components.
    T avg() const;

    //! Returns the minimum value among x, y, and z.
    T min() const;

    //! Returns the maximum value among x, y, and z.
    T max() const;

    //! Returns the absolute minimum value among x, y, and z.
    T absmin() const;

    //! Returns the absolute maximum value among x, y, and z.
    T absmax() const;

    //! Returns the index of the dominant axis.
    size_t dominantAxis() const;

    //! Returns the index of the subminant axis.
    size_t subminantAxis() const;

    //! Returns normalized vector.
    Vector normalized() const;

    //! Returns the length of the vector.
    T length() const;

    //! Returns the squared length of the vector.
    T lengthSquared() const;

    //! Returns the distance to the other vector.
    T distanceTo(const Vector& other) const;

    //! Returns the squared distance to the other vector.
    T distanceSquaredTo(const Vector& other) const;

    //! Returns a vector with different value type.
    template <typename U>
    Vector<U, 4> castTo() const;

    //! Returns true if \p other is the same as this vector.
    bool isEqual(const Vector& other) const;

    //! Returns true if \p other is similar to this vector.
    bool isSimilar(const Vector& other,
                   T epsilon = std::numeric_limits<T>::epsilon()) const;

    // MARK: Operators

    //! Returns reference to the \p i -th element of the vector.
    T& operator[](size_t i);

    //! Returns const reference to the \p i -th element of the vector.
    const T& operator[](size_t i) const;

    //! Set x and y components with given initializer list.
    template <typename U>
    Vector& operator=(const std::initializer_list<U>& lst);

    //! Set x and y with other vector \p v.
    Vector& operator=(const Vector& v);

    //! Computes this += (v, v)
    Vector& operator+=(T v);

    //! Computes this += (v.x, v.y)
    Vector& operator+=(const Vector& v);

    //! Computes this -= (v, v)
    Vector& operator-=(T v);

    //! Computes this -= (v.x, v.y)
    Vector& operator-=(const Vector& v);

    //! Computes this *= (v, v)
    Vector& operator*=(T v);

    //! Computes this *= (v.x, v.y)
    Vector& operator*=(const Vector& v);

    //! Computes this /= (v, v)
    Vector& operator/=(T v);

    //! Computes this /= (v.x, v.y)
    Vector& operator/=(const Vector& v);

    //! Returns true if \p other is the same as this vector.
    bool operator==(const Vector& v) const;

    //! Returns true if \p other is the not same as this vector.
    bool operator!=(const Vector& v) const;
};

//! Type alias for four dimensional vector.
template <typename T>
using Vector4 = Vector<T, 4>;

//! Positive sign operator.
template <typename T>
Vector4<T> operator+(const Vector4<T>& a);

//! Negative sign operator.
template <typename T>
Vector4<T> operator-(const Vector4<T>& a);

//! Computes (a, a, a, a) + (b.x, b.y, b.z, b.w).
template <typename T>
Vector4<T> operator+(T a, const Vector4<T>& b);

//! Computes (a.x, a.y, a.z, a.w) + (b, b, b, b).
template <typename T>
Vector4<T> operator+(const Vector4<T>& a, T b);

//! Computes (a.x, a.y, a.z, a.w) + (b.x, b.y, b.z, b.w).
template <typename T>
Vector4<T> operator+(const Vector4<T>& a, const Vector4<T>& b);

//! Computes (a.x, a.y, a.z, a.w) - (b, b, b, b).
template <typename T>
Vector4<T> operator-(const Vector4<T>& a, T b);

//! Computes (a, a, a, a) - (b.x, b.y, b.z, b.w).
template <typename T>
Vector4<T> operator-(T a, const Vector4<T>& b);

//! Computes (a.x, a.y, a.z, a.w) - (b.x, b.y, b.z, b.w).
template <typename T>
Vector4<T> operator-(const Vector4<T>& a, const Vector4<T>& b);

//! Computes (a.x, a.y, a.z, a.w) * (b, b, b, b).
template <typename T>
Vector4<T> operator*(const Vector4<T>& a, T b);

//! Computes (a, a, a, a) * (b.x, b.y, b.z, b.w).
template <typename T>
Vector4<T> operator*(T a, const Vector4<T>& b);

//! Computes (a.x, a.y, a.z, a.w) * (b.x, b.y, b.z, b.w).
template <typename T>
Vector4<T> operator*(const Vector4<T>& a, const Vector4<T>& b);

//! Computes (a.x, a.y, a.z, a.w) / (b, b, b, b).
template <typename T>
Vector4<T> operator/(const Vector4<T>& a, T b);

//! Computes (a, a, a, a) / (b.x, b.y, b.z, b.w).
template <typename T>
Vector4<T> operator/(T a, const Vector4<T>& b);

//! Computes (a.x, a.y, a.z, a.w) / (b.x, b.y, b.z, b.w).
template <typename T>
Vector4<T> operator/(const Vector4<T>& a, const Vector4<T>& b);

//! Returns element-wise min vector.
template <typename T>
Vector4<T> min(const Vector4<T>& a, const Vector4<T>& b);

//! Returns element-wise max vector.
template <typename T>
Vector4<T> max(const Vector4<T>& a, const Vector4<T>& b);

//! Returns element-wise clamped vector.
template <typename T>
Vector4<T> clamp(const Vector4<T>& v, const Vector4<T>& low,
                 const Vector4<T>& high);

//! Returns element-wise ceiled vector.
template <typename T>
Vector4<T> ceil(const Vector4<T>& a);

//! Returns element-wise floored vector.
template <typename T>
Vector4<T> floor(const Vector4<T>& a);

//! Float-type 4D vector.
typedef Vector4<float> Vector4F;

//! Double-type 4D vector.
typedef Vector4<double> Vector4D;

// MARK: Extensions

//! Returns float-type zero vector.
template <>
constexpr Vector4F zero<Vector4F>() {
    return Vector4F(0.f, 0.f, 0.f, 0.f);
}

//! Returns double-type zero vector.
template <>
constexpr Vector4D zero<Vector4D>() {
    return Vector4D(0.0, 0.0, 0.0, 0.0);
}

//! Returns the type of the value itself.
template <typename T>
struct ScalarType<Vector4<T>> {
    typedef T value;
};

//! Computes monotonic Catmull-Rom interpolation.
template <typename T>
Vector4<T> monotonicCatmullRom(const Vector4<T>& v0, const Vector4<T>& v1,
                               const Vector4<T>& v2, const Vector4<T>& v3, T f);

}  // namespace jet

#include "detail/vector4-inl.h"

#endif  // INCLUDE_JET_VECTOR4_H_
