// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TUPLE_UTILS_H_
#define INCLUDE_JET_TUPLE_UTILS_H_

#include <jet/tuple.h>

namespace jet {

template <typename T, size_t N>
Tuple<T, N> operator+(const Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator+(const T& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator+(const Tuple<T, N>& a, const T& b);

template <typename T, size_t N>
Tuple<T, N> operator+=(Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator+=(Tuple<T, N>& a, const T& b);

template <typename T, size_t N>
Tuple<T, N> operator-(const Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator-(const T& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator-(const Tuple<T, N>& a, const T& b);

template <typename T, size_t N>
Tuple<T, N> operator-=(Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator-=(Tuple<T, N>& a, const T& b);

template <typename T, size_t N>
Tuple<T, N> operator*(const Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator*(const T& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator*(const Tuple<T, N>& a, const T& b);

template <typename T, size_t N>
Tuple<T, N> operator*=(Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator*=(Tuple<T, N>& a, const T& b);

template <typename T, size_t N>
Tuple<T, N> operator/(const Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator/(const T& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator/(const Tuple<T, N>& a, const T& b);

template <typename T, size_t N>
Tuple<T, N> operator/=(Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
Tuple<T, N> operator/=(Tuple<T, N>& a, const T& b);

template <typename T, size_t N>
bool operator==(const Tuple<T, N>& a, const Tuple<T, N>& b);

template <typename T, size_t N>
bool operator!=(const Tuple<T, N>& a, const Tuple<T, N>& b);


//

template <typename T>
Tuple<T, 1> operator+(const Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator+=(Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator+=(Tuple<T, 1>& a, const T& b);

template <typename T>
Tuple<T, 1> operator-(const Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator-(const T& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator-(const Tuple<T, 1>& a, const T& b);

template <typename T>
Tuple<T, 1> operator-=(Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator-=(Tuple<T, 1>& a, const T& b);

template <typename T>
Tuple<T, 1> operator*(const Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator*(const T& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator*(const Tuple<T, 1>& a, const T& b);

template <typename T>
Tuple<T, 1> operator*=(Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator*=(Tuple<T, 1>& a, const T& b);

template <typename T>
Tuple<T, 1> operator/(const Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator/(const T& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator/(const Tuple<T, 1>& a, const T& b);

template <typename T>
Tuple<T, 1> operator/=(Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
Tuple<T, 1> operator/=(Tuple<T, 1>& a, const T& b);

template <typename T>
bool operator==(const Tuple<T, 1>& a, const Tuple<T, 1>& b);

template <typename T>
bool operator!=(const Tuple<T, 1>& a, const Tuple<T, 1>& b);

//

template <typename T>
Tuple<T, 2> operator+(const Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator+(const T& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator+(const Tuple<T, 2>& a, const T& b);

template <typename T>
Tuple<T, 2> operator+=(Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator+=(Tuple<T, 2>& a, const T& b);

template <typename T>
Tuple<T, 2> operator-(const Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator-(const T& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator-(const Tuple<T, 2>& a, const T& b);

template <typename T>
Tuple<T, 2> operator-=(Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator-=(Tuple<T, 2>& a, const T& b);

template <typename T>
Tuple<T, 2> operator*(const Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator*(const T& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator*(const Tuple<T, 2>& a, const T& b);

template <typename T>
Tuple<T, 2> operator*=(Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator*=(Tuple<T, 2>& a, const T& b);

template <typename T>
Tuple<T, 2> operator/(const Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator/(const T& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator/(const Tuple<T, 2>& a, const T& b);

template <typename T>
Tuple<T, 2> operator/=(Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
Tuple<T, 2> operator/=(Tuple<T, 2>& a, const T& b);

template <typename T>
bool operator==(const Tuple<T, 2>& a, const Tuple<T, 2>& b);

template <typename T>
bool operator!=(const Tuple<T, 2>& a, const Tuple<T, 2>& b);

//

template <typename T>
Tuple<T, 3> operator+(const Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator+(const T& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator+(const Tuple<T, 3>& a, const T& b);

template <typename T>
Tuple<T, 3> operator+=(Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator+=(Tuple<T, 3>& a, const T& b);

template <typename T>
Tuple<T, 3> operator-(const Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator-(const T& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator-(const Tuple<T, 3>& a, const T& b);

template <typename T>
Tuple<T, 3> operator-=(Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator-=(Tuple<T, 3>& a, const T& b);

template <typename T>
Tuple<T, 3> operator*(const Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator*(const T& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator*(const Tuple<T, 3>& a, const T& b);

template <typename T>
Tuple<T, 3> operator*=(Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator*=(Tuple<T, 3>& a, const T& b);

template <typename T>
Tuple<T, 3> operator/(const Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator/(const T& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator/(const Tuple<T, 3>& a, const T& b);

template <typename T>
Tuple<T, 3> operator/=(Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
Tuple<T, 3> operator/=(Tuple<T, 3>& a, const T& b);

template <typename T>
bool operator==(const Tuple<T, 3>& a, const Tuple<T, 3>& b);

template <typename T>
bool operator!=(const Tuple<T, 3>& a, const Tuple<T, 3>& b);

//

template <typename T>
Tuple<T, 4> operator+(const Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator+(const T& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator+(const Tuple<T, 4>& a, const T& b);

template <typename T>
Tuple<T, 4> operator+=(Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator+=(Tuple<T, 4>& a, const T& b);

template <typename T>
Tuple<T, 4> operator-(const Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator-(const T& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator-(const Tuple<T, 4>& a, const T& b);

template <typename T>
Tuple<T, 4> operator-=(Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator-=(Tuple<T, 4>& a, const T& b);

template <typename T>
Tuple<T, 4> operator*(const Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator*(const T& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator*(const Tuple<T, 4>& a, const T& b);

template <typename T>
Tuple<T, 4> operator*=(Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator*=(Tuple<T, 4>& a, const T& b);

template <typename T>
Tuple<T, 4> operator/(const Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator/(const T& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator/(const Tuple<T, 4>& a, const T& b);

template <typename T>
Tuple<T, 4> operator/=(Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
Tuple<T, 4> operator/=(Tuple<T, 4>& a, const T& b);

template <typename T>
bool operator==(const Tuple<T, 4>& a, const Tuple<T, 4>& b);

template <typename T>
bool operator!=(const Tuple<T, 4>& a, const Tuple<T, 4>& b);

//

}  // namespace jet

#include <jet/detail/tuple_utils-inl.h>

#endif  // INCLUDE_JET_TUPLE_UTILS_H_
