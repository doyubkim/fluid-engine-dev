// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT2_H_
#define INCLUDE_JET_POINT2_H_

#include <jet/point.h>
#include <algorithm>  // just make cpplint happy..

namespace jet {

template <typename T>
class Point<T, 2> final {
 public:
    static_assert(
        std::is_arithmetic<T>::value,
        "Point only can be instantiated with arithmetic types");

    T x;
    T y;

    // Constructors
    Point();
    Point(T x, T y);
    template <typename U>
    Point(const std::initializer_list<U>& lst);
    Point(const Point& v);

    // Basic setters
    void set(T s);
    void set(T x, T y);
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
    Point<U, 2> castTo() const;

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


template <typename T> using Point2 = Point<T, 2>;

template <typename T>
Point2<T> operator+(const Point2<T>& a);

template <typename T>
Point2<T> operator-(const Point2<T>& a);

template <typename T>
Point2<T> operator+(T a, const Point2<T>& b);

template <typename T>
Point2<T> operator+(const Point2<T>& a, const Point2<T>& b);

template <typename T>
Point2<T> operator-(const Point2<T>& a, T b);

template <typename T>
Point2<T> operator-(T a, const Point2<T>& b);

template <typename T>
Point2<T> operator-(const Point2<T>& a, const Point2<T>& b);

template <typename T>
Point2<T> operator*(const Point2<T>& a, T b);

template <typename T>
Point2<T> operator*(T a, const Point2<T>& b);

template <typename T>
Point2<T> operator*(const Point2<T>& a, const Point2<T>& b);

template <typename T>
Point2<T> operator/(const Point2<T>& a, T b);

template <typename T>
Point2<T> operator/(T a, const Point2<T>& b);

template <typename T>
Point2<T> operator/(const Point2<T>& a, const Point2<T>& b);

template <typename T>
Point2<T> min(const Point2<T>& a, const Point2<T>& b);

template <typename T>
Point2<T> max(const Point2<T>& a, const Point2<T>& b);

template <typename T>
Point2<T> clamp(
    const Point2<T>& v, const Point2<T>& low, const Point2<T>& high);

template <typename T>
Point2<T> ceil(const Point2<T>& a);

template <typename T>
Point2<T> floor(const Point2<T>& a);

typedef Point2<float> Point2F;
typedef Point2<double> Point2D;
typedef Point2<ssize_t> Point2I;
typedef Point2<size_t> Point2UI;

}  // namespace jet

#include "detail/point2-inl.h"

#endif  // INCLUDE_JET_POINT2_H_
