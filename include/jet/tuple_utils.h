// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TUPLE_UTILS_H_
#define INCLUDE_JET_TUPLE_UTILS_H_

#include <jet/tuple.h>

namespace jet {

template <typename T, size_t N, typename D>
constexpr auto operator-(const TupleBase<T, N, D>& a);

template <typename T, size_t N, typename D>
constexpr auto operator+(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator+(const T& a, const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator+(const TupleBase<T, N, D>& a, const T& b);

template <typename T, size_t N, typename D>
constexpr auto operator+=(TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator+=(TupleBase<T, N, D>& a, const T& b);

template <typename T, size_t N, typename D>
constexpr auto operator-(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator-(const T& a, const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator-(const TupleBase<T, N, D>& a, const T& b);

template <typename T, size_t N, typename D>
constexpr auto operator-=(TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator-=(TupleBase<T, N, D>& a, const T& b);

template <typename T, size_t N, typename D>
constexpr auto operator*(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator*(const T& a, const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator*(const TupleBase<T, N, D>& a, const T& b);

template <typename T, size_t N, typename D>
constexpr auto operator*=(TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator*=(TupleBase<T, N, D>& a, const T& b);

template <typename T, size_t N, typename D>
constexpr auto operator/(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator/(const T& a, const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator/(const TupleBase<T, N, D>& a, const T& b);

template <typename T, size_t N, typename D>
constexpr auto operator/=(TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr auto operator/=(TupleBase<T, N, D>& a, const T& b);

template <typename T, size_t N, typename D>
constexpr bool operator==(const TupleBase<T, N, D>& a,
                          const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D>
constexpr bool operator!=(const TupleBase<T, N, D>& a,
                          const TupleBase<T, N, D>& b);

template <typename T, size_t N, typename D, typename BinaryOperation>
constexpr T accumulate(const TupleBase<T, N, D>& a, T init, BinaryOperation op);

template <typename T, size_t N, typename D>
constexpr T accumulate(const TupleBase<T, N, D>& a, T init);

template <typename T, size_t N, typename D>
constexpr T product(const TupleBase<T, N, D>& a, T init);

}  // namespace jet

#include <jet/detail/tuple_utils-inl.h>

#endif  // INCLUDE_JET_TUPLE_UTILS_H_
