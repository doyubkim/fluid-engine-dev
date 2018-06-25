// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_EXPRESSION_H_
#define INCLUDE_JET_MATRIX_EXPRESSION_H_

#include <jet/functors.h>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixExpression

//!
//! \brief Base class for matrix expression.
//!
//! Matrix expression is a meta type that enables template expression pattern.
//!
//! \tparam T  Real number type.
//! \tparam E  Subclass type.
//!
template <typename T, typename Derived>
class MatrixExpression {
 public:
    using value_type = T;

    //! Returns the number of rows.
    constexpr size_t rows() const;

    //! Returns the number of columns.
    constexpr size_t cols() const;

    //! Returns the evaluated value for (i, j).
    T eval(size_t i, size_t j) const;

    //! Returns actual implementation (the subclass).
    Derived& operator()();

    //! Returns actual implementation (the subclass).
    const Derived& operator()() const;

 protected:
    // Prohibits constructing this class instance.
    MatrixExpression() = default;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixConstant

template <typename T>
class MatrixConstant : public MatrixExpression<T, MatrixConstant<T>> {
 public:
    constexpr MatrixConstant(size_t r, size_t c, const T& val)
        : _rows(r), _cols(c), _val(val) {}

    constexpr T rows() const;

    constexpr T cols() const;

    constexpr T operator()(size_t, size_t) const;

 private:
    size_t _rows;
    size_t _cols;
    T _val;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixDiagonal

template <typename T, typename M1>
class MatrixDiagonal : public MatrixExpression<T, MatrixDiagonal<T, M1>> {
 public:
    constexpr MatrixDiagonal(const M1& m1) : _m1(m1) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixOffDiagonal

template <typename T, typename M1>
class MatrixOffDiagonal : public MatrixExpression<T, MatrixOffDiagonal<T, M1>> {
 public:
    constexpr MatrixOffDiagonal(const M1& m1) : _m1(m1) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixTri

template <typename T, typename M1>
class MatrixTri : public MatrixExpression<T, MatrixTri<T, M1>> {
 public:
    constexpr MatrixTri(const M1& m1, bool isUpper, bool isStrict)
        : _m1(m1), _isUpper(isUpper), _isStrict(isStrict) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
    bool _isUpper;
    bool _isStrict;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixTranspose

template <typename T, typename M1>
class MatrixTranspose : public MatrixExpression<T, MatrixTranspose<T, M1>> {
 public:
    constexpr MatrixTranspose(const M1& m1) : _m1(m1) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixUnaryOp

template <typename T, typename M1, typename UnaryOperation>
class MatrixUnaryOp
    : public MatrixExpression<T, MatrixUnaryOp<T, M1, UnaryOperation>> {
 public:
    constexpr MatrixUnaryOp(const M1& m1) : _m1(m1) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
    UnaryOperation _op;
};

template <typename T, typename M1>
using MatrixNegate = MatrixUnaryOp<T, M1, std::negate<T>>;

template <typename T, typename M1>
using MatrixCeil = MatrixUnaryOp<T, M1, Ceil<T>>;

template <typename T, typename M1>
using MatrixFloor = MatrixUnaryOp<T, M1, Floor<T>>;

template <typename T, typename U, typename M1>
using MatrixTypeCast = MatrixUnaryOp<U, M1, TypeCast<T, U>>;

template <typename T, typename M1>
constexpr auto ceil(const MatrixExpression<T, M1>& a);

template <typename T, typename M1>
constexpr auto floor(const MatrixExpression<T, M1>& a);

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto operator-(const MatrixExpression<T, M1>& m);

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixElemWiseBinaryOp

//!
//! \brief Matrix expression for element-wise binary operation.
//!
//! This matrix expression represents a binary matrix operation that takes
//! two input matrix expressions.
//!
//! \tparam T                   Real number type.
//! \tparam E1                  First input expression type.
//! \tparam E2                  Second input expression type.
//! \tparam BinaryOperation     Binary operation.
//!
template <typename T, typename E1, typename E2, typename BinaryOperation>
class MatrixElemWiseBinaryOp
    : public MatrixExpression<
          T, MatrixElemWiseBinaryOp<T, E1, E2, BinaryOperation>> {
 public:
    constexpr MatrixElemWiseBinaryOp(const E1& m1, const E2& m2)
        : _m1(m1), _m2(m2) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr T operator()(size_t i, size_t j) const;

 private:
    E1 _m1;
    E2 _m2;
    BinaryOperation _op;
};

//! Matrix expression for element-wise matrix-matrix addition.
template <typename T, typename E1, typename E2>
using MatrixElemWiseAdd = MatrixElemWiseBinaryOp<T, E1, E2, std::plus<T>>;

//! Matrix expression for element-wise matrix-matrix subtraction.
template <typename T, typename E1, typename E2>
using MatrixElemWiseSub = MatrixElemWiseBinaryOp<T, E1, E2, std::minus<T>>;

//! Matrix expression for element-wise matrix-matrix multiplication.
template <typename T, typename E1, typename E2>
using MatrixElemWiseMul = MatrixElemWiseBinaryOp<T, E1, E2, std::multiplies<T>>;

//! Matrix expression for element-wise matrix-matrix division.
template <typename T, typename E1, typename E2>
using MatrixElemWiseDiv = MatrixElemWiseBinaryOp<T, E1, E2, std::divides<T>>;

//! Matrix expression for element-wise matrix-matrix min operation.
template <typename T, typename E1, typename E2>
using MatrixElemWiseMin = MatrixElemWiseBinaryOp<T, E1, E2, Min<T>>;

//! Matrix expression for element-wise matrix-matrix max operation.
template <typename T, typename E1, typename E2>
using MatrixElemWiseMax = MatrixElemWiseBinaryOp<T, E1, E2, Max<T>>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M1, typename M2>
constexpr auto operator+(const MatrixExpression<T, M1>& a,
                         const MatrixExpression<T, M2>& b);

template <typename T, typename M1, typename M2>
constexpr auto operator-(const MatrixExpression<T, M1>& a,
                         const MatrixExpression<T, M2>& b);

template <typename T, typename M1, typename M2>
constexpr auto elemMul(const MatrixExpression<T, M1>& a,
                       const MatrixExpression<T, M2>& b);

template <typename T, typename M1, typename M2>
constexpr auto elemDiv(const MatrixExpression<T, M1>& a,
                       const MatrixExpression<T, M2>& b);

template <typename T, typename M1, typename M2>
constexpr auto min(const MatrixExpression<T, M1>& a,
                   const MatrixExpression<T, M2>& b);

template <typename T, typename M1, typename M2>
constexpr auto max(const MatrixExpression<T, M1>& a,
                   const MatrixExpression<T, M2>& b);

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixScalarElemWiseBinaryOp

template <typename T, typename M1, typename BinaryOperation>
class MatrixScalarElemWiseBinaryOp
    : public MatrixExpression<
          T, MatrixScalarElemWiseBinaryOp<T, M1, BinaryOperation>> {
 public:
    constexpr MatrixScalarElemWiseBinaryOp(const M1& m1, const T& s2)
        : _m1(m1), _s2(s2) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
    T _s2;
    BinaryOperation _op;
};

template <typename T, typename M1>
using MatrixScalarElemWiseAdd =
    MatrixScalarElemWiseBinaryOp<T, M1, std::plus<T>>;

template <typename T, typename M1>
using MatrixScalarElemWiseSub =
    MatrixScalarElemWiseBinaryOp<T, M1, std::minus<T>>;

template <typename T, typename M1>
using MatrixScalarElemWiseMul =
    MatrixScalarElemWiseBinaryOp<T, M1, std::multiplies<T>>;

template <typename T, typename M1>
using MatrixScalarElemWiseDiv =
    MatrixScalarElemWiseBinaryOp<T, M1, std::divides<T>>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M1>
constexpr auto operator+(const MatrixExpression<T, M1>& a, const T& b);

template <typename T, typename M1>
constexpr auto operator-(const MatrixExpression<T, M1>& a, const T& b);

template <typename T, typename M1>
constexpr auto operator*(const MatrixExpression<T, M1>& a, const T& b);

template <typename T, typename M1>
constexpr auto operator/(const MatrixExpression<T, M1>& a, const T& b);

////////////////////////////////////////////////////////////////////////////////
// MARK: ScalarMatrixElemWiseBinaryOp

template <typename T, typename M2, typename BinaryOperation>
class ScalarMatrixElemWiseBinaryOp
    : public MatrixExpression<
          T, ScalarMatrixElemWiseBinaryOp<T, M2, BinaryOperation>> {
 public:
    constexpr ScalarMatrixElemWiseBinaryOp(const T& s1, const M2& m2)
        : _s1(s1), _m2(m2) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr T operator()(size_t i, size_t j) const;

 private:
    T _s1;
    M2 _m2;
    BinaryOperation _op;
};

template <typename T, typename M2>
using ScalarMatrixElemWiseAdd =
    ScalarMatrixElemWiseBinaryOp<T, M2, std::plus<T>>;

template <typename T, typename M2>
using ScalarMatrixElemWiseSub =
    ScalarMatrixElemWiseBinaryOp<T, M2, std::minus<T>>;

template <typename T, typename M2>
using ScalarMatrixElemWiseMul =
    ScalarMatrixElemWiseBinaryOp<T, M2, std::multiplies<T>>;

template <typename T, typename M2>
using ScalarMatrixElemWiseDiv =
    ScalarMatrixElemWiseBinaryOp<T, M2, std::divides<T>>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M2>
constexpr auto operator+(const T& a, const MatrixExpression<T, M2>& b);

template <typename T, typename M2>
constexpr auto operator-(const T& a, const MatrixExpression<T, M2>& b);

template <typename T, typename M2>
constexpr auto operator*(const T& a, const MatrixExpression<T, M2>& b);

template <typename T, typename M2>
constexpr auto operator/(const T& a, const MatrixExpression<T, M2>& b);

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixTernaryOp

template <typename T, typename M1, typename M2, typename M3,
          typename TernaryOperation>
class MatrixTernaryOp
    : public MatrixExpression<
          T, MatrixTernaryOp<T, M1, M2, M3, TernaryOperation>> {
 public:
    constexpr MatrixTernaryOp(const M1& m1, const M2& m2, const M3& m3)
        : _m1(m1), _m2(m2), _m3(m3) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
    M2 _m2;
    M3 _m3;
    TernaryOperation _op;
};

template <typename T, typename M1, typename M2, typename M3>
using MatrixClamp = MatrixTernaryOp<T, M1, M2, M3, Clamp<T>>;

template <typename T, typename M1, typename M2, typename M3>
constexpr auto clamp(const MatrixExpression<T, M1>& a,
                     const MatrixExpression<T, M2>& low,
                     const MatrixExpression<T, M3>& high);

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixMul

template <typename T, typename M1, typename M2>
class MatrixMul : public MatrixExpression<T, MatrixMul<T, M1, M2>> {
 public:
    constexpr MatrixMul(const M1& m1, const M2& m2) : _m1(m1), _m2(m2) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
    M2 _m2;
};

template <typename T, typename M1, typename M2>
constexpr auto operator*(const MatrixExpression<T, M1>& a,
                         const MatrixExpression<T, M2>& b);

}  // namespace jet

#include <jet/detail/matrix_expression-inl.h>

#endif  // INCLUDE_JET_MATRIX_EXPRESSION_H_
