// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT3_H_
#define INCLUDE_JET_POINT3_H_

#include <jet/point2.h>
#include <algorithm>  // just make cpplint happy..

namespace jet {

//!
//! \brief 3-D point class.
//!
//! This class defines simple 3-D point data.
//!
//! \tparam T - Type of the element
//!
template <typename T>
class Point<T, 3> final {
 public:
    static_assert(std::is_arithmetic<T>::value,
                  "Point only can be instantiated with arithematic types");

    //! X (or the first) component of the point.
    T x;

    //! Y (or the second) component of the point.
    T y;

    //! Z (or the third) component of the point.
    T z;

    // MARK: Constructors

    //! Constructs default point (0, 0, 0).
    constexpr Point() : x(0), y(0), z(0) {}

    //! Constructs point with given parameters \p x_, \p y_, and \p z_.
    constexpr Point(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    //! Constructs point with a 2-D point and a scalar.
    constexpr Point(const Point2<T>& v, T z_) : x(v.x), y(v.y), z(z_) {}

    //! Constructs point with initializer list.
    template <typename U>
    Point(const std::initializer_list<U>& lst);

    //! Copy constructor.
    constexpr Point(const Point& v) : x(v.x), y(v.y), z(v.z) {}

    // MARK: Basic setters

    //! Set all x, y, and z components to \p s.
    void set(T s);

    //! Set x, y, and z components with given parameters.
    void set(T x, T y, T z);

    //! Set x, y, and z components with given \p pt.x, \p pt.y, and \p z.
    void set(const Point2<T>& pt, T z);

    //! Set x, y, and z components with given initializer list.
    template <typename U>
    void set(const std::initializer_list<U>& lst);

    //! Set x, y, and z with other point \p pt.
    void set(const Point& v);

    //! Set both x, y, and z to zero.
    void setZero();

    // MARK: Binary operations: new instance = this (+) v

    //! Computes this + (v, v, v).
    Point add(T v) const;

    //! Computes this + (v.x, v.y, v.z).
    Point add(const Point& v) const;

    //! Computes this - (v, v, v).
    Point sub(T v) const;

    //! Computes this - (v.x, v.y, v.z).
    Point sub(const Point& v) const;

    //! Computes this * (v, v, v).
    Point mul(T v) const;

    //! Computes this * (v.x, v.y, v.z).
    Point mul(const Point& v) const;
    //! Computes this / (v, v, v).
    Point div(T v) const;

    //! Computes this / (v.x, v.y, v.z).
    Point div(const Point& v) const;

    // MARK: Binary operations: new instance = v (+) this

    //! Computes (v, v, v) - this.
    Point rsub(T v) const;

    //! Computes (v.x, v.y, v.z) - this.
    Point rsub(const Point& v) const;

    //! Computes (v, v, v) / this.
    Point rdiv(T v) const;

    //! Computes (v.x, v.y, v.z) / this.
    Point rdiv(const Point& v) const;

    // MARK: Augmented operations: this (+)= v

    //! Computes this += (v, v, v).
    void iadd(T v);

    //! Computes this += (v.x, v.y, v.z).
    void iadd(const Point& v);

    //! Computes this -= (v, v, v).
    void isub(T v);

    //! Computes this -= (v.x, v.y, v.z).
    void isub(const Point& v);

    //! Computes this *= (v, v, v).
    void imul(T v);

    //! Computes this *= (v.x, v.y, v.z).
    void imul(const Point& v);

    //! Computes this /= (v, v, v).
    void idiv(T v);

    //! Computes this /= (v.x, v.y, v.z).
    void idiv(const Point& v);

    // MARK: Basic getters

    //! Returns const reference to the \p i -th element of the point.
    const T& at(size_t i) const;

    //! Returns reference to the \p i -th element of the point.
    T& at(size_t i);

    //! Returns the sum of all the components (i.e. x + y).
    T sum() const;

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

    //! Returns a point with different value type.
    template <typename U>
    Point<U, 3> castTo() const;

    //! Returns true if \p other is the same as this point.
    bool isEqual(const Point& other) const;

    // MARK: Operators

    //! Returns reference to the \p i -th element of the point.
    T& operator[](size_t i);

    //! Returns const reference to the \p i -th element of the point.
    const T& operator[](size_t i) const;

    //! Set x, y, and z components with given initializer list.
    Point& operator=(const std::initializer_list<T>& lst);

    //! Set x, y, and z with other point \p pt.
    Point& operator=(const Point& v);

    //! Computes this += (v, v, v)
    Point& operator+=(T v);

    //! Computes this += (v.x, v.y, v.z)
    Point& operator+=(const Point& v);

    //! Computes this -= (v, v, v)
    Point& operator-=(T v);

    //! Computes this -= (v.x, v.y, v.z)
    Point& operator-=(const Point& v);

    //! Computes this *= (v, v, v)
    Point& operator*=(T v);

    //! Computes this *= (v.x, v.y, v.z)
    Point& operator*=(const Point& v);

    //! Computes this /= (v, v, v)
    Point& operator/=(T v);

    //! Computes this /= (v.x, v.y, v.z)
    Point& operator/=(const Point& v);

    //! Returns true if \p other is the same as this point.
    bool operator==(const Point& v) const;

    //! Returns true if \p other is the not same as this point.
    bool operator!=(const Point& v) const;
};

//! Type alias for three dimensional point.
template <typename T>
using Point3 = Point<T, 3>;

//! Positive sign operator.
template <typename T>
Point<T, 3> operator+(const Point<T, 3>& a);

//! Negative sign operator.
template <typename T>
Point3<T> operator-(const Point3<T>& a);

//! Computes (a, a, a) + (b.x, b.y, b.z).
template <typename T>
Point3<T> operator+(T a, const Point3<T>& b);

//! Computes (a.x, a.y, a.z) + (b.x, b.y, b.z).
template <typename T>
Point3<T> operator+(const Point3<T>& a, const Point3<T>& b);

//! Computes (a.x, a.y, a.z) - (b, b, b).
template <typename T>
Point3<T> operator-(const Point3<T>& a, T b);

//! Computes (a, a, a) - (b.x, b.y, b.z).
template <typename T>
Point3<T> operator-(T a, const Point3<T>& b);

//! Computes (a.x, a.y, a.z) - (b.x, b.y, b.z).
template <typename T>
Point3<T> operator-(const Point3<T>& a, const Point3<T>& b);

//! Computes (a.x, a.y, a.z) * (b, b, b).
template <typename T>
Point3<T> operator*(const Point3<T>& a, T b);

//! Computes (a, a, a) * (b.x, b.y, b.z).
template <typename T>
Point3<T> operator*(T a, const Point3<T>& b);

//! Computes (a.x, a.y, a.z) * (b.x, b.y, b.z).
template <typename T>
Point3<T> operator*(const Point3<T>& a, const Point3<T>& b);

//! Computes (a.x, a.y, a.z) / (b, b, b).
template <typename T>
Point3<T> operator/(const Point3<T>& a, T b);

//! Computes (a, a, a) / (b.x, b.y, b.z).
template <typename T>
Point3<T> operator/(T a, const Point3<T>& b);

//! Computes (a.x, a.y, a.z) / (b.x, b.y, b.z).
template <typename T>
Point3<T> operator/(const Point3<T>& a, const Point3<T>& b);

//! Returns element-wise min point.
template <typename T>
Point3<T> min(const Point3<T>& a, const Point3<T>& b);

//! Returns element-wise max point.
template <typename T>
Point3<T> max(const Point3<T>& a, const Point3<T>& b);

//! Returns element-wise clamped point.
template <typename T>
Point3<T> clamp(const Point3<T>& v, const Point3<T>& low,
                const Point3<T>& high);

//! Returns element-wise ceiled point.
template <typename T>
Point3<T> ceil(const Point3<T>& a);

//! Returns element-wise floored point.
template <typename T>
Point3<T> floor(const Point3<T>& a);

//! Float-type 3D point.
typedef Point3<float> Point3F;

//! Double-type 3D point.
typedef Point3<double> Point3D;

//! Integer-type 3D point.
typedef Point3<ssize_t> Point3I;

//! Unsigned integer-type 3D point.
typedef Point3<size_t> Point3UI;

}  // namespace jet

#include "detail/point3-inl.h"

#endif  // INCLUDE_JET_POINT3_H_
