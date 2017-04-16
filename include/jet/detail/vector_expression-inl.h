// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_VECTOR_EXPRESSION_INL_H
#define INCLUDE_JET_DETAIL_VECTOR_EXPRESSION_INL_H

#include <jet/vector_expression.h>

namespace jet {

// MARK: VectorExpression

template <typename T, typename E>
size_t VectorExpression<T, E>::size() const {
    return static_cast<const E&>(*this).size();
}

template <typename T, typename E>
const E& VectorExpression<T, E>::operator()() const {
    return static_cast<const E&>(*this);
}

// MARK: VectorUnaryOp

template <typename T, typename E, typename Op>
VectorUnaryOp<T, E, Op>::VectorUnaryOp(const E& u) : _u(u) {}

template <typename T, typename E, typename Op>
size_t VectorUnaryOp<T, E, Op>::size() const {
    return _u.size();
}

template <typename T, typename E, typename Op>
T VectorUnaryOp<T, E, Op>::operator[](size_t i) const {
    return _op(_u[i]);
}

// MARK: VectorBinaryOp

template <typename T, typename E1, typename E2, typename Op>
VectorBinaryOp<T, E1, E2, Op>::VectorBinaryOp(const E1& u, const E2& v)
    : _u(u), _v(v) {
    JET_ASSERT(u.size() == v.size());
}

template <typename T, typename E1, typename E2, typename Op>
size_t VectorBinaryOp<T, E1, E2, Op>::size() const {
    return _v.size();
}

template <typename T, typename E1, typename E2, typename Op>
T VectorBinaryOp<T, E1, E2, Op>::operator[](size_t i) const {
    return _op(_u[i], _v[i]);
}

template <typename T, typename E, typename Op>
VectorScalarBinaryOp<T, E, Op>::VectorScalarBinaryOp(const E& u, const T& v)
    : _u(u), _v(v) {}

template <typename T, typename E, typename Op>
size_t VectorScalarBinaryOp<T, E, Op>::size() const {
    return _u.size();
}

template <typename T, typename E, typename Op>
T VectorScalarBinaryOp<T, E, Op>::operator[](size_t i) const {
    return _op(_u[i], _v);
}

// MARK: Global Functions

template <typename T, typename E>
VectorScalarAdd<T, E> operator+(const T& a, const VectorExpression<T, E>& b) {
    return VectorScalarAdd<T, E>(b(), a);
}

template <typename T, typename E>
VectorScalarAdd<T, E> operator+(const VectorExpression<T, E>& a, const T& b) {
    return VectorScalarAdd<T, E>(a(), b);
}

template <typename T, typename E1, typename E2>
VectorAdd<T, E1, E2> operator+(const VectorExpression<T, E1>& a,
                               const VectorExpression<T, E2>& b) {
    return VectorAdd<T, E1, E2>(a(), b());
}

template <typename T, typename E>
VectorScalarRSub<T, E> operator-(const T& a, const VectorExpression<T, E>& b) {
    return VectorScalarRSub<T, E>(b(), a);
}

template <typename T, typename E>
VectorScalarSub<T, E> operator-(const VectorExpression<T, E>& a, const T& b) {
    return VectorScalarSub<T, E>(a(), b);
}

template <typename T, typename E1, typename E2>
VectorSub<T, E1, E2> operator-(const VectorExpression<T, E1>& a,
                               const VectorExpression<T, E2>& b) {
    return VectorSub<T, E1, E2>(a(), b());
}

template <typename T, typename E>
VectorScalarMul<T, E> operator*(const T& a, const VectorExpression<T, E>& b) {
    return VectorScalarMul<T, E>(b(), a);
}

template <typename T, typename E>
VectorScalarMul<T, E> operator*(const VectorExpression<T, E>& a, const T& b) {
    return VectorScalarMul<T, E>(a(), b);
}

template <typename T, typename E1, typename E2>
VectorMul<T, E1, E2> operator*(const VectorExpression<T, E1>& a,
                               const VectorExpression<T, E2>& b) {
    return VectorMul<T, E1, E2>(a(), b());
}

template <typename T, typename E>
VectorScalarRDiv<T, E> operator/(const T& a, const VectorExpression<T, E>& b) {
    return VectorScalarRDiv<T, E>(b(), a);
}

template <typename T, typename E>
VectorScalarDiv<T, E> operator/(const VectorExpression<T, E>& a, const T& b) {
    return VectorScalarDiv<T, E>(a(), b);
}

template <typename T, typename E1, typename E2>
VectorDiv<T, E1, E2> operator/(const VectorExpression<T, E1>& a,
                               const VectorExpression<T, E2>& b) {
    return VectorDiv<T, E1, E2>(a(), b());
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_VECTOR_EXPRESSION_INL_H
