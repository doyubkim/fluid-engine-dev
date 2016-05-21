// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT3_H_
#define INCLUDE_JET_POINT3_H_

#include <jet/point2.h>
#include <algorithm>  // just make cpplint happy..

namespace jet {

template <typename T>
class Point<T, 3> final {
 public:
    static_assert(
        std::is_arithmetic<T>::value,
        "Point only can be instantiated with arithematic types");

    T x;
    T y;
    T z;

    // Constructors
    Point();
    Point(T x, T y, T z);
    Point(const Point2<T>& pt, T z);
    template <typename U>
    Point(const std::initializer_list<U>& lst);
    Point(const Point& v);

    // Basic setters
    void set(T s);
    void set(T x, T y, T z);
    void set(const Point2<T>& pt, T z);
    template <typename U>
    void set(const std::initializer_list<U>& lst);
    void set(const Point& v);

    void setZero();

    // Binary operations: new instance = this (+) v
    Point add(T v) const;
    Point add(const Point& v) const;
    Point sub(T v) const;
    Point sub(const Point& v) const;
    Point mul(T v) const;
    Point mul(const Point& v) const;
    Point div(T v) const;
    Point div(const Point& v) const;

    // Binary operations: new instance = v (+) this
    Point rsub(T v) const;
    Point rsub(const Point& v) const;
    Point rdiv(T v) const;
    Point rdiv(const Point& v) const;

    // Augmented operations: this (+)= v
    void iadd(T v);
    void iadd(const Point& v);
    void isub(T v);
    void isub(const Point& v);
    void imul(T v);
    void imul(const Point& v);
    void idiv(T v);
    void idiv(const Point& v);

    // Basic getters
    const T& at(size_t i) const;
    T& at(size_t i);
    T sum() const;
    T min() const;
    T max() const;
    T absmin() const;
    T absmax() const;
    size_t dominantAxis() const;
    size_t subminantAxis() const;

    template <typename U>
    Point<U, 3> castTo() const;

    bool isEqual(const Point& other) const;

    // Operators
    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    Point& operator=(const std::initializer_list<T>& lst);
    Point& operator=(const Point& v);
    Point& operator+=(T v);
    Point& operator+=(const Point& v);
    Point& operator-=(T v);
    Point& operator-=(const Point& v);
    Point& operator*=(T v);
    Point& operator*=(const Point& v);
    Point& operator/=(T v);
    Point& operator/=(const Point& v);

    bool operator==(const Point& v) const;
    bool operator!=(const Point& v) const;
};


template <typename T> using Point3 = Point<T, 3>;

template <typename T>
Point3<T> operator+(const Point3<T>& a);

template <typename T>
Point3<T> operator-(const Point3<T>& a);

template <typename T>
Point3<T> operator+(T a, const Point3<T>& b);

template <typename T>
Point3<T> operator+(const Point3<T>& a, const Point3<T>& b);

template <typename T>
Point3<T> operator-(const Point3<T>& a, T b);

template <typename T>
Point3<T> operator-(T a, const Point3<T>& b);

template <typename T>
Point3<T> operator-(const Point3<T>& a, const Point3<T>& b);

template <typename T>
Point3<T> operator*(const Point3<T>& a, T b);

template <typename T>
Point3<T> operator*(T a, const Point3<T>& b);

template <typename T>
Point3<T> operator*(const Point3<T>& a, const Point3<T>& b);

template <typename T>
Point3<T> operator/(const Point3<T>& a, T b);

template <typename T>
Point3<T> operator/(T a, const Point3<T>& b);

template <typename T>
Point3<T> operator/(const Point3<T>& a, const Point3<T>& b);

template <typename T>
Point3<T> min(const Point3<T>& a, const Point3<T>& b);

template <typename T>
Point3<T> max(const Point3<T>& a, const Point3<T>& b);

template <typename T>
Point3<T> clamp(
    const Point3<T>& v, const Point3<T>& low, const Point3<T>& high);

template <typename T>
Point3<T> ceil(const Point3<T>& a);

template <typename T>
Point3<T> floor(const Point3<T>& a);

typedef Point3<float> Point3F;
typedef Point3<double> Point3D;
typedef Point3<ssize_t> Point3I;
typedef Point3<size_t> Point3UI;

}  // namespace jet

#include "detail/point3-inl.h"

#endif  // INCLUDE_JET_POINT3_H_

