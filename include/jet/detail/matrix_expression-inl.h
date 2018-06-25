// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_MATRIX_EXPRESSION_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX_EXPRESSION_INL_H_

#include <jet/matrix_expression.h>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixExpression

template <typename T, typename D>
constexpr size_t MatrixExpression<T, D>::rows() const {
    return static_cast<const D&>(*this).rows();
}

template <typename T, typename D>
constexpr size_t MatrixExpression<T, D>::cols() const {
    return static_cast<const D&>(*this).cols();
}

template <typename T, typename D>
T MatrixExpression<T, D>::eval(size_t i, size_t j) const {
    return (*this)()(i, j);
}

template <typename T, typename D>
D& MatrixExpression<T, D>::operator()() {
    return static_cast<D&>(*this);
}

template <typename T, typename D>
const D& MatrixExpression<T, D>::operator()() const {
    return static_cast<const D&>(*this);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixConstant

template <typename T>
constexpr T MatrixConstant<T>::rows() const {
    return _rows;
}

template <typename T>
constexpr T MatrixConstant<T>::cols() const {
    return _cols;
}

template <typename T>
constexpr T MatrixConstant<T>::operator()(size_t, size_t) const {
    return _val;
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixDiagonal

template <typename T, typename M1>
constexpr size_t MatrixDiagonal<T, M1>::rows() const {
    return _m1.rows();
}

template <typename T, typename M1>
constexpr size_t MatrixDiagonal<T, M1>::cols() const {
    return _m1.cols();
}

template <typename T, typename M1>
T MatrixDiagonal<T, M1>::operator()(size_t i, size_t j) const {
    if (i == j) {
        return _m1(i, j);
    } else {
        return T{};
    }
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixOffDiagonal

template <typename T, typename M1>
constexpr size_t MatrixOffDiagonal<T, M1>::rows() const {
    return _m1.rows();
}

template <typename T, typename M1>
constexpr size_t MatrixOffDiagonal<T, M1>::cols() const {
    return _m1.cols();
}

template <typename T, typename M1>
T MatrixOffDiagonal<T, M1>::operator()(size_t i, size_t j) const {
    if (i != j) {
        return _m1(i, j);
    } else {
        return T{};
    }
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixTri

template <typename T, typename M1>
constexpr size_t MatrixTri<T, M1>::rows() const {
    return _m1.rows();
}

template <typename T, typename M1>
constexpr size_t MatrixTri<T, M1>::cols() const {
    return _m1.cols();
}

template <typename T, typename M1>
T MatrixTri<T, M1>::operator()(size_t i, size_t j) const {
    if (_isUpper) {
        if (_isStrict) {
            return (j > i) ? _m1(i, j) : 0;
        } else {
            return (j >= i) ? _m1(i, j) : 0;
        }
    } else {
        if (_isStrict) {
            return (j < i) ? _m1(i, j) : 0;
        } else {
            return (j <= i) ? _m1(i, j) : 0;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixTranspose

template <typename T, typename M1>
constexpr size_t MatrixTranspose<T, M1>::rows() const {
    return _m1.cols();
}

template <typename T, typename M1>
constexpr size_t MatrixTranspose<T, M1>::cols() const {
    return _m1.rows();
}

template <typename T, typename M1>
constexpr T MatrixTranspose<T, M1>::operator()(size_t i, size_t j) const {
    return _m1(j, i);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixUnaryOp

template <typename T, typename M1, typename UOp>
constexpr size_t MatrixUnaryOp<T, M1, UOp>::rows() const {
    return _m1.rows();
}

template <typename T, typename M1, typename UOp>
constexpr size_t MatrixUnaryOp<T, M1, UOp>::cols() const {
    return _m1.cols();
}

template <typename T, typename M1, typename UOp>
constexpr T MatrixUnaryOp<T, M1, UOp>::operator()(size_t i, size_t j) const {
    return _op(_m1(i, j));
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M1>
constexpr auto ceil(const MatrixExpression<T, M1>& a) {
    return MatrixCeil<T, const M1&>{a()};
}

template <typename T, typename M1>
constexpr auto floor(const MatrixExpression<T, M1>& a) {
    return MatrixFloor<T, const M1&>{a()};
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto operator-(const MatrixExpression<T, M1>& m) {
    return MatrixNegate<T, const M1&>{m()};
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixElemWiseBinaryOp

template <typename T, typename E1, typename E2, typename BOp>
constexpr size_t MatrixElemWiseBinaryOp<T, E1, E2, BOp>::rows() const {
    return _m1.rows();
}

template <typename T, typename E1, typename E2, typename BOp>
constexpr size_t MatrixElemWiseBinaryOp<T, E1, E2, BOp>::cols() const {
    return _m1.cols();
}

template <typename T, typename E1, typename E2, typename BOp>
constexpr T MatrixElemWiseBinaryOp<T, E1, E2, BOp>::operator()(size_t i,
                                                               size_t j) const {
    return _op(_m1(i, j), _m2(i, j));
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M1, typename M2>
constexpr auto operator+(const MatrixExpression<T, M1>& a,
                         const MatrixExpression<T, M2>& b) {
    return MatrixElemWiseAdd<T, const M1&, const M2&>{a(), b()};
}

template <typename T, typename M1, typename M2>
constexpr auto operator-(const MatrixExpression<T, M1>& a,
                         const MatrixExpression<T, M2>& b) {
    return MatrixElemWiseSub<T, const M1&, const M2&>{a(), b()};
}

template <typename T, typename M1, typename M2>
constexpr auto elemMul(const MatrixExpression<T, M1>& a,
                       const MatrixExpression<T, M2>& b) {
    return MatrixElemWiseMul<T, const M1&, const M2&>{a(), b()};
}

template <typename T, typename M1, typename M2>
constexpr auto elemDiv(const MatrixExpression<T, M1>& a,
                       const MatrixExpression<T, M2>& b) {
    return MatrixElemWiseDiv<T, const M1&, const M2&>{a(), b()};
}

template <typename T, typename M1, typename M2>
constexpr auto min(const MatrixExpression<T, M1>& a,
                   const MatrixExpression<T, M2>& b) {
    return MatrixElemWiseMin<T, const M1&, const M2&>{a(), b()};
}

template <typename T, typename M1, typename M2>
constexpr auto max(const MatrixExpression<T, M1>& a,
                   const MatrixExpression<T, M2>& b) {
    return MatrixElemWiseMax<T, const M1&, const M2&>{a(), b()};
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixScalarElemWiseBinaryOp

template <typename T, typename M1, typename BOp>
constexpr size_t MatrixScalarElemWiseBinaryOp<T, M1, BOp>::rows() const {
    return _m1.rows();
}

template <typename T, typename M1, typename BOp>
constexpr size_t MatrixScalarElemWiseBinaryOp<T, M1, BOp>::cols() const {
    return _m1.cols();
}

template <typename T, typename M1, typename BOp>
constexpr T MatrixScalarElemWiseBinaryOp<T, M1, BOp>::operator()(
    size_t i, size_t j) const {
    return _op(_m1(i, j), _s2);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M1>
constexpr auto operator+(const MatrixExpression<T, M1>& a, const T& b) {
    return MatrixScalarElemWiseAdd<T, const M1&>{a(), b};
}

template <typename T, typename M1>
constexpr auto operator-(const MatrixExpression<T, M1>& a, const T& b) {
    return MatrixScalarElemWiseSub<T, const M1&>{a(), b};
}

template <typename T, typename M1>
constexpr auto operator*(const MatrixExpression<T, M1>& a, const T& b) {
    return MatrixScalarElemWiseMul<T, const M1&>{a(), b};
}

template <typename T, typename M1>
constexpr auto operator/(const MatrixExpression<T, M1>& a, const T& b) {
    return MatrixScalarElemWiseDiv<T, const M1&>{a(), b};
}

////////////////////////////////////////////////////////////////////////////////
// MARK: ScalarMatrixElemWiseBinaryOp

template <typename T, typename M2, typename BOp>
constexpr size_t ScalarMatrixElemWiseBinaryOp<T, M2, BOp>::rows() const {
    return _m2.rows();
}

template <typename T, typename M2, typename BOp>
constexpr size_t ScalarMatrixElemWiseBinaryOp<T, M2, BOp>::cols() const {
    return _m2.cols();
}

template <typename T, typename M2, typename BOp>
constexpr T ScalarMatrixElemWiseBinaryOp<T, M2, BOp>::operator()(
    size_t i, size_t j) const {
    return _op(_s1, _m2(i, j));
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M2>
constexpr auto operator+(const T& a, const MatrixExpression<T, M2>& b) {
    return ScalarMatrixElemWiseAdd<T, const M2&>{a, b()};
}

template <typename T, typename M2>
constexpr auto operator-(const T& a, const MatrixExpression<T, M2>& b) {
    return ScalarMatrixElemWiseSub<T, const M2&>{a, b()};
}

template <typename T, typename M2>
constexpr auto operator*(const T& a, const MatrixExpression<T, M2>& b) {
    return ScalarMatrixElemWiseMul<T, const M2&>{a, b()};
}

template <typename T, typename M2>
constexpr auto operator/(const T& a, const MatrixExpression<T, M2>& b) {
    return ScalarMatrixElemWiseDiv<T, const M2&>{a, b()};
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixTernaryOp

template <typename T, typename M1, typename M2, typename M3, typename TOp>
constexpr size_t MatrixTernaryOp<T, M1, M2, M3, TOp>::rows() const {
    return _m1.rows();
}

template <typename T, typename M1, typename M2, typename M3, typename TOp>
constexpr size_t MatrixTernaryOp<T, M1, M2, M3, TOp>::cols() const {
    return _m1.cols();
}

template <typename T, typename M1, typename M2, typename M3, typename TOp>
constexpr T MatrixTernaryOp<T, M1, M2, M3, TOp>::operator()(size_t i,
                                                            size_t j) const {
    return _op(_m1(i, j), _m2(i, j), _m3(i, j));
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M1, typename M2, typename M3>
constexpr auto clamp(const MatrixExpression<T, M1>& a,
                     const MatrixExpression<T, M2>& low,
                     const MatrixExpression<T, M3>& high) {
    JET_ASSERT(a.rows() == low.rows() && a.rows() == high.rows());
    JET_ASSERT(a.cols() == low.cols() && a.cols() == high.cols());
    return MatrixClamp<T, const M1&, const M2&, const M3&>{a(), low(), high()};
}

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixMul

template <typename T, typename M1, typename M2>
constexpr size_t MatrixMul<T, M1, M2>::rows() const {
    return _m1.rows();
}

template <typename T, typename M1, typename M2>
constexpr size_t MatrixMul<T, M1, M2>::cols() const {
    return _m2.cols();
}

template <typename T, typename M1, typename M2>
T MatrixMul<T, M1, M2>::operator()(size_t i, size_t j) const {
    T sum = _m1(i, 0) * _m2(0, j);
    for (size_t k = 1; k < _m1.cols(); ++k) {
        sum += _m1(i, k) * _m2(k, j);
    }
    return sum;
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M1, typename M2>
constexpr auto operator*(const MatrixExpression<T, M1>& a,
                         const MatrixExpression<T, M2>& b) {
    return MatrixMul<T, const M1&, const M2&>{a(), b()};
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_MATRIX_EXPRESSION_INL_H_
