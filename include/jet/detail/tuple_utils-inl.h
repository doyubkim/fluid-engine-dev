// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_TUPLE_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_TUPLE_UTILS_INL_H_

#include <jet/constants.h>
#include <jet/functors.h>
#include <jet/tuple_utils.h>

#include <functional>

namespace jet {

namespace internal {

template <typename... Args>
void noOp(Args...) {}

template <typename T>
T assign(T& a, const T& b) {
    return a = b;
}

template <typename T, typename U>
U takeSecond(const T&, const U& b) {
    return b;
}

// TODO: With C++17, fold expression could be used instead.
template <typename T, size_t N, typename D, typename ReduceOperation, size_t I>
struct Reduce {
    constexpr static auto call(const TupleBase<T, N, D>& a, const T& init,
                               ReduceOperation op) {
        return op(Reduce<T, N, D, ReduceOperation, I - 1>::call(a, init, op),
                  a[I]);
    }
};

template <typename T, size_t N, typename D, typename ReduceOperation>
struct Reduce<T, N, D, ReduceOperation, 0> {
    constexpr static auto call(const TupleBase<T, N, D>& a, const T& init,
                               ReduceOperation op) {
        return op(a[0], init);
    }
};

// We can use std::logical_and<>, but explicitly putting && helps compiler to
// early terminate the loop (at least for gcc 8.1 as I checked the assembly).
// TODO: With C++17, fold expression could be used instead.
template <typename T, size_t N, typename D, typename BinaryOperation, size_t I>
struct FoldWithAnd {
    constexpr static bool call(const TupleBase<T, N, D>& a,
                               const TupleBase<T, N, D>& b,
                               BinaryOperation op) {
        return FoldWithAnd<T, N, D, BinaryOperation, I - 1>::call(a, b, op) &&
               op(a[I], b[I]);
    }
};

template <typename T, size_t N, typename D, typename BinaryOperation>
struct FoldWithAnd<T, N, D, BinaryOperation, 0> {
    constexpr static bool call(const TupleBase<T, N, D>& a,
                               const TupleBase<T, N, D>& b,
                               BinaryOperation op) {
        return op(a[0], b[0]);
    }
};

//

template <typename T, size_t N, typename D, size_t... I>
constexpr auto makeTuple(const T& val, std::index_sequence<I...>) {
    return D{takeSecond(I, val)...};
}

//

template <typename T, size_t N, typename D, typename Op, size_t... I>
constexpr auto unaryOp(const TupleBase<T, N, D>& a, Op op,
                       std::index_sequence<I...>) {
    return D{op(a[I])...};
}

template <typename T, size_t N, typename D, typename Op,
          typename Indices = std::make_index_sequence<N>>
constexpr auto unaryOp(const TupleBase<T, N, D>& a, Op op) {
    return unaryOp(a, op, Indices{});
}

//

template <typename T, size_t N, typename D, typename Op, size_t... I>
constexpr auto binaryOp(const TupleBase<T, N, D>& a,
                        const TupleBase<T, N, D>& b, Op op,
                        std::index_sequence<I...>) {
    return D{op(a[I], b[I])...};
}

template <typename T, size_t N, typename D, typename Op>
auto binaryOp(const TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b, Op op) {
    using Indices = std::make_index_sequence<N>;
    return binaryOp(a, b, op, Indices{});
}

template <typename T, size_t N, typename D, typename Op>
auto binaryOp(const T& a, const TupleBase<T, N, D>& b, Op op) {
    using Indices = std::make_index_sequence<N>;
    return binaryOp(makeTuple<T, N, D>(a, Indices{}), b, op, Indices{});
}

template <typename T, size_t N, typename D, typename Op>
constexpr auto binaryOp(const TupleBase<T, N, D>& a, const T& b, Op op) {
    using Indices = std::make_index_sequence<N>;
    return binaryOp(a, makeTuple<T, N, D>(b, Indices{}), op, Indices{});
}

//

template <typename T, size_t N, typename D, typename Op, size_t... I>
constexpr auto ternaryOp(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b,
                         const TupleBase<T, N, D>& c, Op op,
                         std::index_sequence<I...>) {
    return D{op(a[I], b[I], c[I])...};
}

template <typename T, size_t N, typename D, typename Op>
auto ternaryOp(const TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b,
               const TupleBase<T, N, D>& c, Op op) {
    using Indices = std::make_index_sequence<N>;
    return ternaryOp(a, b, c, op, Indices{});
}

//

template <typename T, size_t N, typename D, size_t... I>
void fill(TupleBase<T, N, D>& a, const T& val, std::index_sequence<I...>) {
    noOp(assign(a[I], val)...);
}

}  // namespace internal

// MARK: Basic Operators

template <typename T, size_t N, typename D>
constexpr auto operator-(const TupleBase<T, N, D>& a) {
    return internal::unaryOp(a, std::negate<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator+(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, std::plus<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator+(const T& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, std::plus<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator+(const TupleBase<T, N, D>& a, const T& b) {
    return internal::binaryOp(a, b, std::plus<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator+=(TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, IAdd<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator+=(TupleBase<T, N, D>& a, const T& b) {
    return internal::binaryOp(a, b, IAdd<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator-(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, std::minus<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator-(const T& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, std::minus<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator-(const TupleBase<T, N, D>& a, const T& b) {
    return internal::binaryOp(a, b, std::minus<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator-=(TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, ISub<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator-=(TupleBase<T, N, D>& a, const T& b) {
    return internal::binaryOp(a, b, ISub<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator*(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, std::multiplies<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator*(const T& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, std::multiplies<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator*(const TupleBase<T, N, D>& a, const T& b) {
    return internal::binaryOp(a, b, std::multiplies<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator*=(TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, IMul<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator*=(TupleBase<T, N, D>& a, const T& b) {
    return internal::binaryOp(a, b, IMul<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator/(const TupleBase<T, N, D>& a,
                         const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, std::divides<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator/(const T& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, std::divides<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator/(const TupleBase<T, N, D>& a, const T& b) {
    return internal::binaryOp(a, b, std::divides<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator/=(TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, IDiv<T>{});
}

template <typename T, size_t N, typename D>
constexpr auto operator/=(TupleBase<T, N, D>& a, const T& b) {
    return internal::binaryOp(a, b, IDiv<T>{});
}

template <typename T, size_t N, typename D>
constexpr bool operator==(const TupleBase<T, N, D>& a,
                          const TupleBase<T, N, D>& b) {
    return internal::FoldWithAnd<T, N, D, std::equal_to<T>, N - 1>::call(
        a, b, std::equal_to<T>());
}

template <typename T, size_t N, typename D>
constexpr bool operator!=(const TupleBase<T, N, D>& a,
                          const TupleBase<T, N, D>& b) {
    return !(a == b);
}

// MARK: Simple Utilities

template <typename T, size_t N, typename D>
void fill(TupleBase<T, N, D>& a, const T& val) {
    internal::fill(a, val, std::make_index_sequence<N>{});
}

template <typename T, size_t N, typename D, typename BinaryOperation>
constexpr T accumulate(const TupleBase<T, N, D>& a, const T& init,
                       BinaryOperation op) {
    return internal::Reduce<T, N, D, BinaryOperation, N - 1>::call(a, init, op);
}

template <typename T, size_t N, typename D>
constexpr T accumulate(const TupleBase<T, N, D>& a, const T& init) {
    return internal::Reduce<T, N, D, std::plus<T>, N - 1>::call(a, init,
                                                                std::plus<T>());
}

template <typename T, size_t N, typename D>
constexpr T product(const TupleBase<T, N, D>& a, const T& init) {
    return accumulate(a, init, std::multiplies<T>());
}

template <typename T, size_t N, typename D>
constexpr auto min(const TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, Min<T>());
}

template <typename T, size_t N, typename D>
constexpr auto max(const TupleBase<T, N, D>& a, const TupleBase<T, N, D>& b) {
    return internal::binaryOp(a, b, Max<T>());
}

template <typename T, size_t N, typename D>
constexpr auto clamp(const TupleBase<T, N, D>& a, const TupleBase<T, N, D>& low,
                     const TupleBase<T, N, D>& high) {
    return internal::ternaryOp(a, low, high, Clamp<T>());
}

template <typename T, size_t N, typename D>
constexpr auto ceil(const TupleBase<T, N, D>& a) {
    return internal::unaryOp(a, Ceil<T>());
}

template <typename T, size_t N, typename D>
constexpr auto floor(const TupleBase<T, N, D>& a) {
    return internal::unaryOp(a, Floor<T>());
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_TUPLE_UTILS_INL_H_
