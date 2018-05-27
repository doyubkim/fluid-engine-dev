// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_TUPLE_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_TUPLE_UTILS_INL_H_

#include <jet/functors.h>
#include <jet/tuple_utils.h>

#include <functional>

namespace jet {

namespace internal {

template <typename T, size_t N, typename Op, size_t... I>
Tuple<T, N> binaryOp(const Tuple<T, N>& a, const Tuple<T, N>& b, const Op& op,
                     std::index_sequence<I...>) {
    return Tuple<T, N>{op(a[I], b[I])...};
}

template <typename T, size_t N, typename Op,
          typename Indices = std::make_index_sequence<N>>
Tuple<T, N> binaryOp(const Tuple<T, N>& a, const Tuple<T, N>& b, const Op& op) {
    return internal::binaryOp(a, b, op, Indices{});
}

template <typename T, size_t N, typename Op, size_t... I>
Tuple<T, N> binaryOp(const T& a, const Tuple<T, N>& b, const Op& op,
                     std::index_sequence<I...>) {
    return Tuple<T, N>{op(a, b[I])...};
}

template <typename T, size_t N, typename Op,
          typename Indices = std::make_index_sequence<N>>
Tuple<T, N> binaryOp(const T& a, const Tuple<T, N>& b, const Op& op) {
    return internal::binaryOp(a, b, op, Indices{});
}

template <typename T, size_t N, typename Op, size_t... I>
Tuple<T, N> binaryOp(const Tuple<T, N>& a, const T& b, const Op& op,
                     std::index_sequence<I...>) {
    return Tuple<T, N>{op(a[I], b)...};
}

template <typename T, size_t N, typename Op,
          typename Indices = std::make_index_sequence<N>>
Tuple<T, N> binaryOp(const Tuple<T, N>& a, const T& b, const Op& op) {
    return internal::binaryOp(a, b, op, Indices{});
}

}  // namespace internal

template <typename T, size_t N>
Tuple<T, N> operator+(const Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::plus<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator+(const T& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::plus<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator+(const Tuple<T, N>& a, const T& b) {
    return internal::binaryOp(a, b, std::plus<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator+=(Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, IAdd<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator+=(Tuple<T, N>& a, const T& b) {
    return internal::binaryOp(a, b, IAdd<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator-(const Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::minus<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator-(const T& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::minus<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator-(const Tuple<T, N>& a, const T& b) {
    return internal::binaryOp(a, b, std::minus<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator-=(Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, ISub<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator-=(Tuple<T, N>& a, const T& b) {
    return internal::binaryOp(a, b, ISub<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator*(const Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::multiplies<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator*(const T& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::multiplies<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator*(const Tuple<T, N>& a, const T& b) {
    return internal::binaryOp(a, b, std::multiplies<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator*=(Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, IMul<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator*=(Tuple<T, N>& a, const T& b) {
    return internal::binaryOp(a, b, IMul<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator/(const Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::divides<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator/(const T& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::divides<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator/(const Tuple<T, N>& a, const T& b) {
    return internal::binaryOp(a, b, std::divides<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator/=(Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, IDiv<T>{});
}

template <typename T, size_t N>
Tuple<T, N> operator/=(Tuple<T, N>& a, const T& b) {
    return internal::binaryOp(a, b, IDiv<T>{});
}

template <typename T, size_t N>
bool operator==(const Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::equal_to<T>{});
}

template <typename T, size_t N>
bool operator!=(const Tuple<T, N>& a, const Tuple<T, N>& b) {
    return internal::binaryOp(a, b, std::not_equal_to<T>{});
}

//

template <typename T>
Tuple<T, 1> operator+(const Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    return Tuple<T, 1>{a.x + b.x};
}

template <typename T>
Tuple<T, 1> operator+(const T& a, const Tuple<T, 1>& b) {
    return Tuple<T, 1>{a + b.x};
}

template <typename T>
Tuple<T, 1> operator+(const Tuple<T, 1>& a, const T& b) {
    return Tuple<T, 1>{a.x + b};
}

template <typename T>
Tuple<T, 1> operator+=(Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    a.x += b.x;
}

template <typename T>
Tuple<T, 1> operator+=(Tuple<T, 1>& a, const T& b) {
    a.x += b;
}

template <typename T>
Tuple<T, 1> operator-(const Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    return Tuple<T, 1>{a.x - b.x};
}

template <typename T>
Tuple<T, 1> operator-(const T& a, const Tuple<T, 1>& b) {
    return Tuple<T, 1>{a - b.x};
}

template <typename T>
Tuple<T, 1> operator-(const Tuple<T, 1>& a, const T& b) {
    return Tuple<T, 1>{a.x - b};
}

template <typename T>
Tuple<T, 1> operator-=(Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    a.x -= b.x;
}

template <typename T>
Tuple<T, 1> operator-=(Tuple<T, 1>& a, const T& b) {
    a.x -= b;
}

template <typename T>
Tuple<T, 1> operator*(const Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    return Tuple<T, 1>{a.x * b.x};
}

template <typename T>
Tuple<T, 1> operator*(const T& a, const Tuple<T, 1>& b) {
    return Tuple<T, 1>{a * b.x};
}

template <typename T>
Tuple<T, 1> operator*(const Tuple<T, 1>& a, const T& b) {
    return Tuple<T, 1>{a.x * b};
}

template <typename T>
Tuple<T, 1> operator*=(Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    a.x *= b.x;
}

template <typename T>
Tuple<T, 1> operator*=(Tuple<T, 1>& a, const T& b) {
    a.x *= b;
}

template <typename T>
Tuple<T, 1> operator/(const Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    return Tuple<T, 1>{a.x / b.x};
}

template <typename T>
Tuple<T, 1> operator/(const T& a, const Tuple<T, 1>& b) {
    return Tuple<T, 1>{a / b.x};
}

template <typename T>
Tuple<T, 1> operator/(const Tuple<T, 1>& a, const T& b) {
    return Tuple<T, 1>{a.x / b};
}

template <typename T>
Tuple<T, 1> operator/=(Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    a.x /= b.x;
}

template <typename T>
Tuple<T, 1> operator/=(Tuple<T, 1>& a, const T& b) {
    a.x /= b;
}

template <typename T>
bool operator==(const Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    return a.x == b.x;
}

template <typename T>
bool operator!=(const Tuple<T, 1>& a, const Tuple<T, 1>& b) {
    return !(a == b);
}

//

template <typename T>
Tuple<T, 2> operator+(const Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    return Tuple<T, 2>{a.x + b.x, a.y + b.y};
}

template <typename T>
Tuple<T, 2> operator+(const T& a, const Tuple<T, 2>& b) {
    return Tuple<T, 2>{a + b.x, a + b.y};
}

template <typename T>
Tuple<T, 2> operator+(const Tuple<T, 2>& a, const T& b) {
    return Tuple<T, 2>{a.x + b, a.y + b};
}

template <typename T>
Tuple<T, 2> operator+=(Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    a.x += b.x;
    a.y += b.y;
}

template <typename T>
Tuple<T, 2> operator+=(Tuple<T, 2>& a, T& b) {
    a.x += b;
    a.y += b;
}

template <typename T>
Tuple<T, 2> operator-(const Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    return Tuple<T, 2>{a.x - b.x, a.y - b.y};
}

template <typename T>
Tuple<T, 2> operator-(const T& a, const Tuple<T, 2>& b) {
    return Tuple<T, 2>{a - b.x, a - b.y};
}

template <typename T>
Tuple<T, 2> operator-(const Tuple<T, 2>& a, const T& b) {
    return Tuple<T, 2>{a.x - b, a.y - b};
}

template <typename T>
Tuple<T, 2> operator-=(Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    a.x -= b.x;
    a.y -= b.y;
}

template <typename T>
Tuple<T, 2> operator-=(Tuple<T, 2>& a, T& b) {
    a.x -= b;
    a.y -= b;
}

template <typename T>
Tuple<T, 2> operator*(const Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    return Tuple<T, 2>{a.x * b.x, a.y * b.y};
}

template <typename T>
Tuple<T, 2> operator*(const T& a, const Tuple<T, 2>& b) {
    return Tuple<T, 2>{a * b.x, a * b.y};
}

template <typename T>
Tuple<T, 2> operator*(const Tuple<T, 2>& a, const T& b) {
    return Tuple<T, 2>{a.x * b, a.y * b};
}

template <typename T>
Tuple<T, 2> operator*=(Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    a.x *= b.x;
    a.y *= b.y;
}

template <typename T>
Tuple<T, 2> operator*=(Tuple<T, 2>& a, T& b) {
    a.x *= b;
    a.y *= b;
}

template <typename T>
Tuple<T, 2> operator/(const Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    return Tuple<T, 2>{a.x / b.x, a.y / b.y};
}

template <typename T>
Tuple<T, 2> operator/(const T& a, const Tuple<T, 2>& b) {
    return Tuple<T, 2>{a / b.x, a / b.y};
}

template <typename T>
Tuple<T, 2> operator/(const Tuple<T, 2>& a, const T& b) {
    return Tuple<T, 2>{a.x / b, a.y / b};
}

template <typename T>
Tuple<T, 2> operator/=(Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    a.x /= b.x;
    a.y /= b.y;
}

template <typename T>
Tuple<T, 2> operator/=(Tuple<T, 2>& a, T& b) {
    a.x /= b;
    a.y /= b;
}

template <typename T>
bool operator==(const Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    return a.x == b.x && a.y == b.y;
}

template <typename T>
bool operator!=(const Tuple<T, 2>& a, const Tuple<T, 2>& b) {
    return !(a == b);
}

//

template <typename T>
Tuple<T, 3> operator+(const Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    return Tuple<T, 3>{a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename T>
Tuple<T, 3> operator+(const T& a, const Tuple<T, 3>& b) {
    return Tuple<T, 3>{a + b.x, a + b.y, a + b.z};
}

template <typename T>
Tuple<T, 3> operator+(const Tuple<T, 3>& a, const T& b) {
    return Tuple<T, 3>{a.x + b, a.y + b, a.z + b};
}

template <typename T>
Tuple<T, 3> operator+=(Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

template <typename T>
Tuple<T, 3> operator+=(Tuple<T, 3>& a, const T& b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

template <typename T>
Tuple<T, 3> operator-(const Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    return Tuple<T, 3>{a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename T>
Tuple<T, 3> operator-(const T& a, const Tuple<T, 3>& b) {
    return Tuple<T, 3>{a - b.x, a - b.y, a - b.z};
}

template <typename T>
Tuple<T, 3> operator-(const Tuple<T, 3>& a, const T& b) {
    return Tuple<T, 3>{a.x - b, a.y - b, a.z - b};
}

template <typename T>
Tuple<T, 3> operator-=(Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

template <typename T>
Tuple<T, 3> operator-=(Tuple<T, 3>& a, const T& b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

template <typename T>
Tuple<T, 3> operator*(const Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    return Tuple<T, 3>{a.x * b.x, a.y * b.y, a.z * b.z};
}

template <typename T>
Tuple<T, 3> operator*(const T& a, const Tuple<T, 3>& b) {
    return Tuple<T, 3>{a * b.x, a * b.y, a * b.z};
}

template <typename T>
Tuple<T, 3> operator*(const Tuple<T, 3>& a, const T& b) {
    return Tuple<T, 3>{a.x * b, a.y * b, a.z * b};
}

template <typename T>
Tuple<T, 3> operator*=(Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

template <typename T>
Tuple<T, 3> operator*=(Tuple<T, 3>& a, const T& b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

template <typename T>
Tuple<T, 3> operator/(const Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    return Tuple<T, 3>{a.x / b.x, a.y / b.y, a.z / b.z};
}

template <typename T>
Tuple<T, 3> operator/(const T& a, const Tuple<T, 3>& b) {
    return Tuple<T, 3>{a / b.x, a / b.y, a / b.z};
}

template <typename T>
Tuple<T, 3> operator/(const Tuple<T, 3>& a, const T& b) {
    return Tuple<T, 3>{a.x / b, a.y / b, a.z / b};
}

template <typename T>
Tuple<T, 3> operator/=(Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

template <typename T>
Tuple<T, 3> operator/=(Tuple<T, 3>& a, const T& b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

template <typename T>
bool operator==(const Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

template <typename T>
bool operator!=(const Tuple<T, 3>& a, const Tuple<T, 3>& b) {
    return !(a == b);
}

//

template <typename T>
Tuple<T, 4> operator+(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    return Tuple<T, 4>{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <typename T>
Tuple<T, 4> operator+(const T& a, const Tuple<T, 4>& b) {
    return Tuple<T, 4>{a + b.x, a + b.y, a + b.z, a + b.w};
}

template <typename T>
Tuple<T, 4> operator+(const Tuple<T, 4>& a, const T& b) {
    return Tuple<T, 4>{a.x + b, a.y + b, a.z + b, a.w + b};
}

template <typename T>
Tuple<T, 4> operator+=(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

template <typename T>
Tuple<T, 4> operator+=(const Tuple<T, 4>& a, const T& b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

template <typename T>
Tuple<T, 4> operator-(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    return Tuple<T, 4>{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template <typename T>
Tuple<T, 4> operator-(const T& a, const Tuple<T, 4>& b) {
    return Tuple<T, 4>{a - b.x, a - b.y, a - b.z, a - b.w};
}

template <typename T>
Tuple<T, 4> operator-(const Tuple<T, 4>& a, const T& b) {
    return Tuple<T, 4>{a.x - b, a.y - b, a.z - b, a.w - b};
}

template <typename T>
Tuple<T, 4> operator-=(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

template <typename T>
Tuple<T, 4> operator-=(const Tuple<T, 4>& a, const T& b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

template <typename T>
Tuple<T, 4> operator*(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    return Tuple<T, 4>{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <typename T>
Tuple<T, 4> operator*(const T& a, const Tuple<T, 4>& b) {
    return Tuple<T, 4>{a * b.x, a * b.y, a * b.z, a * b.w};
}

template <typename T>
Tuple<T, 4> operator*(const Tuple<T, 4>& a, const T& b) {
    return Tuple<T, 4>{a.x * b, a.y * b, a.z * b, a.w * b};
}

template <typename T>
Tuple<T, 4> operator*=(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

template <typename T>
Tuple<T, 4> operator*=(const Tuple<T, 4>& a, const T& b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

template <typename T>
Tuple<T, 4> operator/(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    return Tuple<T, 4>{a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

template <typename T>
Tuple<T, 4> operator/(const T& a, const Tuple<T, 4>& b) {
    return Tuple<T, 4>{a / b.x, a / b.y, a / b.z, a / b.w};
}

template <typename T>
Tuple<T, 4> operator/(const Tuple<T, 4>& a, const T& b) {
    return Tuple<T, 4>{a.x / b, a.y / b, a.z / b, a.w / b};
}

template <typename T>
Tuple<T, 4> operator/=(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

template <typename T>
Tuple<T, 4> operator/=(const Tuple<T, 4>& a, const T& b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

template <typename T>
bool operator==(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <typename T>
bool operator!=(const Tuple<T, 4>& a, const Tuple<T, 4>& b) {
    return !(a == b);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_TUPLE_UTILS_INL_H_
