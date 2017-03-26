// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VECTOR2_H_
#define INCLUDE_JET_VECTOR2_H_

#include <jet/vector.h>
#include <algorithm>  // just make cpplint happy..
#include <limits>

namespace jet {

//!
//! \brief 2-D vector class.
//!
//! This class defines simple 2-D vector data.
//!
//! \tparam T - Type of the element
//!
template <typename T>
class Vector<T, 2> final {
 public:
    static_assert(std::is_floating_point<T>::value,
                  "Vector only can be instantiated with floating point types");

    //! X (or the first) component of the vector.
    T x;

    //! Y (or the second) component of the vector.
    T y;

    // MARK: Constructors

    //! Constructs default vector (0, 0).
    constexpr Vector() : x(0), y(0) {}

    //! Constructs vector with given parameters \p x_ and \p y_.
    constexpr Vector(T x_, T y_) : x(x_), y(y_) {}

    //! Constructs vector with initializer list.
    template <typename U>
    Vector(const std::initializer_list<U>& lst);

    //! Copy constructor.
    constexpr Vector(const Vector& v) : x(v.x), y(v.y) {}

    // MARK: Basic setters

    //! Set both x and y components to \p s.
    void set(T s);

    //! Set x and y components with given parameters.
    void set(T x, T y);

    //! Set x and y components with given initializer list.
    template <typename U>
    void set(const std::initializer_list<U>& lst);

    //! Set x and y with other vector \p pt.
    void set(const Vector& pt);

    //! Set both x and y to zero.
    void setZero();

    //! Normalizes this vector.
    void normalize();

    // MARK: Binary operations: new instance = this (+) v

    //! Computes this + (v, v).
    Vector add(T v) const;

    //! Computes this + (v.x, v.y).
    Vector add(const Vector& v) const;

    //! Computes this - (v, v).
    Vector sub(T v) const;

    //! Computes this - (v.x, v.y).
    Vector sub(const Vector& v) const;

    //! Computes this * (v, v).
    Vector mul(T v) const;

    //! Computes this * (v.x, v.y).
    Vector mul(const Vector& v) const;

    //! Computes this / (v, v).
    Vector div(T v) const;

    //! Computes this / (v.x, v.y).
    Vector div(const Vector& v) const;

    //! Computes dot product.
    T dot(const Vector& v) const;

    //! Comptues cross product.
    T cross(const Vector& v) const;

    // MARK: Binary operations: new instance = v (+) this

    //! Computes (v, v) - this.
    Vector rsub(T v) const;

    //! Computes (v.x, v.y) - this.
    Vector rsub(const Vector& v) const;

    //! Computes (v, v) / this.
    Vector rdiv(T v) const;

    //! Computes (v.x, v.y) / this.
    Vector rdiv(const Vector& v) const;

    //! Computes \p v cross this.
    T rcross(const Vector& v) const;

    // MARK: Augmented operations: this (+)= v

    //! Computes this += (v, v).
    void iadd(T v);

    //! Computes this += (v.x, v.y).
    void iadd(const Vector& v);

    //! Computes this -= (v, v).
    void isub(T v);

    //! Computes this -= (v.x, v.y).
    void isub(const Vector& v);

    //! Computes this *= (v, v).
    void imul(T v);

    //! Computes this *= (v.x, v.y).
    void imul(const Vector& v);

    //! Computes this /= (v, v).
    void idiv(T v);

    //! Computes this /= (v.x, v.y).
    void idiv(const Vector& v);

    // MARK: Basic getters

    //! Returns const reference to the \p i -th element of the vector.
    const T& at(size_t i) const;

    //! Returns reference to the \p i -th element of the vector.
    T& at(size_t i);

    //! Returns the sum of all the components (i.e. x + y).
    T sum() const;

    //! Returns the average of all the components.
    T avg() const;

    //! Returns the minimum value among x and y.
    T min() const;

    //! Returns the maximum value among x and y.
    T max() const;

    //! Returns the absolute minimum value among x and y.
    T absmin() const;

    //! Returns the absolute maximum value among x and y.
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

    //! Returns the reflection vector to the surface with given surface normal.
    Vector reflected(const Vector& normal) const;

    //! Returns the projected vector to the surface with given surface normal.
    Vector projected(const Vector& normal) const;

    //! Returns the tangential vector for this vector.
    Vector tangential() const;

    //! Returns a vector with different value type.
    template <typename U>
    Vector<U, 2> castTo() const;

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

    //! Set x and y with other vector \p pt.
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

//! Type alias for two dimensional vector.
template <typename T>
using Vector2 = Vector<T, 2>;

//! Positive sign operator.
template <typename T>
Vector2<T> operator+(const Vector2<T>& a);

//! Negative sign operator.
template <typename T>
Vector2<T> operator-(const Vector2<T>& a);

//! Computes (a, a) + (b.x, b.y).
template <typename T>
Vector2<T> operator+(T a, const Vector2<T>& b);

//! Computes (a.x, a.y) + (b.x, b.y).
template <typename T>
Vector2<T> operator+(const Vector2<T>& a, const Vector2<T>& b);

//! Computes (a.x, a.y) - (b, b).
template <typename T>
Vector2<T> operator-(const Vector2<T>& a, T b);

//! Computes (a, a) - (b.x, b.y).
template <typename T>
Vector2<T> operator-(T a, const Vector2<T>& b);

//! Computes (a.x, a.y) - (b.x, b.y).
template <typename T>
Vector2<T> operator-(const Vector2<T>& a, const Vector2<T>& b);

//! Computes (a.x, a.y) * (b, b).
template <typename T>
Vector2<T> operator*(const Vector2<T>& a, T b);

//! Computes (a, a) * (b.x, b.y).
template <typename T>
Vector2<T> operator*(T a, const Vector2<T>& b);

//! Computes (a.x, a.y) * (b.x, b.y).
template <typename T>
Vector2<T> operator*(const Vector2<T>& a, const Vector2<T>& b);

//! Computes (a.x, a.y) / (b, b).
template <typename T>
Vector2<T> operator/(const Vector2<T>& a, T b);

//! Computes (a, a) / (b.x, b.y).
template <typename T>
Vector2<T> operator/(T a, const Vector2<T>& b);

//! Computes (a.x, a.y) / (b.x, b.y).
template <typename T>
Vector2<T> operator/(const Vector2<T>& a, const Vector2<T>& b);

//! Returns element-wise min vector: (min(a.x, b.x), min(a.y, b.y)).
template <typename T>
Vector2<T> min(const Vector2<T>& a, const Vector2<T>& b);

//! Returns element-wise max vector: (max(a.x, b.x), max(a.y, b.y)).
template <typename T>
Vector2<T> max(const Vector2<T>& a, const Vector2<T>& b);

//! Returns element-wise clamped vector.
template <typename T>
Vector2<T> clamp(const Vector2<T>& v, const Vector2<T>& low,
                 const Vector2<T>& high);

//! Returns element-wise ceiled vector.
template <typename T>
Vector2<T> ceil(const Vector2<T>& a);

//! Returns element-wise floored vector.
template <typename T>
Vector2<T> floor(const Vector2<T>& a);

//! Float-type 2D vector.
typedef Vector2<float> Vector2F;

//! Double-type 2D vector.
typedef Vector2<double> Vector2D;

// MARK: Extensions

//! Returns float-type zero vector.
template <>
constexpr Vector2F zero<Vector2F>() {
    return Vector2F(0.f, 0.f);
}

//! Returns double-type zero vector.
template <>
constexpr Vector2D zero<Vector2D>() {
    return Vector2D(0.0, 0.0);
}

//! Returns the type of the value itself.
template <typename T>
struct ScalarType<Vector2<T>> {
    typedef T value;
};

//! Computes monotonic Catmull-Rom interpolation.
template <typename T>
Vector2<T> monotonicCatmullRom(const Vector2<T>& v0, const Vector2<T>& v1,
                               const Vector2<T>& v2, const Vector2<T>& v3, T f);

}  // namespace jet

#include "detail/vector2-inl.h"

#endif  // INCLUDE_JET_VECTOR2_H_
