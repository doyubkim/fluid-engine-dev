// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_EXPRESSION_H_
#define INCLUDE_JET_MATRIX_EXPRESSION_H_

#include <jet/size2.h>
#include <jet/vector_expression.h>

namespace jet {

// MARK: MatrixExpression

//!
//! \brief Base class for matrix expression.
//!
//! Matrix expression is a meta type that enables template expression pattern.
//!
//! \tparam T  Real number type.
//! \tparam E  Subclass type.
//!
template <typename T, typename E>
class MatrixExpression {
 public:
    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns actual implementation (the subclass).
    const E& operator()() const;
};

//!
//! \brief Constant matrix expression.
//!
//! This matrix expression represents a constant matrix.
//!
//! \tparam T  Real number type.
//!
template <typename T>
class MatrixConstant : public MatrixExpression<T, MatrixConstant<T>> {
 public:
    //! Constructs m x n constant matrix expression.
    MatrixConstant(size_t m, size_t n, const T& c);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    size_t _m;
    size_t _n;
    T _c;
};

//!
//! \brief Identity matrix expression.
//!
//! This matrix expression represents an identity matrix.
//!
//! \tparam T  Real number type.
//!
template <typename T>
class MatrixIdentity : public MatrixExpression<T, MatrixIdentity<T>> {
 public:
    //! Constructs m x m identity matrix expression.
    MatrixIdentity(size_t m);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    size_t _m;
};

// MARK: MatrixUnaryOp

//!
//! \brief Matrix expression for unary operation.
//!
//! This matrix expression represents an unary matrix operation that takes
//! single input matrix expression.
//!
//! \tparam T   Real number type.
//! \tparam E   Input expression type.
//! \tparam Op  Unary operation.
//!
template <typename T, typename E, typename Op>
class MatrixUnaryOp : public MatrixExpression<T, MatrixUnaryOp<T, E, Op>> {
 public:
    //! Constructs unary operation expression for given input expression.
    MatrixUnaryOp(const E& u);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    const E& _u;
    Op _op;
};

//!
//! \brief Diagonal matrix expression.
//!
//! This matrix expression represents a diagonal matrix for given input matrix
//! expression.
//!
//! \tparam T  Real number type.
//! \tparam E  Input expression type.
//!
template <typename T, typename E>
class MatrixDiagonal : public MatrixExpression<T, MatrixDiagonal<T, E>> {
 public:
    //! Constructs diagonal matrix expression for given input expression.
    //! \param isDiag - True for diagonal matrix, false for off-diagonal.
    MatrixDiagonal(const E& u, bool isDiag);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    const E& _u;
    bool _isDiag;
};

//!
//! \brief Triangular matrix expression.
//!
//! This matrix expression represents a triangular matrix for given input matrix
//! expression.
//!
//! \tparam T  Real number type.
//! \tparam E  Input expression type.
//!
template <typename T, typename E>
class MatrixTriangular : public MatrixExpression<T, MatrixTriangular<T, E>> {
 public:
    //! Constructs triangular matrix expression for given input expression.
    //! \param isUpper - True for upper tri matrix, false for lower tri matrix.
    //! \param isStrict - True for strictly upper/lower triangular matrix.
    MatrixTriangular(const E& u, bool isUpper, bool isStrict);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    const E& _u;
    bool _isUpper;
    bool _isStrict;
};

// MARK: MatrixUnaryOp Aliases

//! Matrix expression for type casting.
template <typename T, typename E, typename U>
using MatrixTypeCast = MatrixUnaryOp<T, E, TypeCast<U, T>>;

// MARK: MatrixBinaryOp

//!
//! \brief Matrix expression for binary operation.
//!
//! This matrix expression represents a binary matrix operation that takes
//! two input matrix expressions.
//!
//! \tparam T   Real number type.
//! \tparam E1  First input expression type.
//! \tparam E2  Second input expression type.
//! \tparam Op  Binary operation.
//!
template <typename T, typename E1, typename E2, typename Op>
class MatrixBinaryOp
    : public MatrixExpression<T, MatrixBinaryOp<T, E1, E2, Op>> {
 public:
    //! Constructs binary operation expression for given input matrix
    //! expressions.
    MatrixBinaryOp(const E1& u, const E2& v);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    const E1& _u;
    const E2& _v;
    Op _op;
};

//!
//! \brief Matrix expression for matrix-scalar binary operation.
//!
//! This matrix expression represents a binary matrix operation that takes
//! one input matrix expression and one scalar.
//!
//! \tparam T   Real number type.
//! \tparam E   Input expression type.
//! \tparam Op  Binary operation.
//!
template <typename T, typename E, typename Op>
class MatrixScalarBinaryOp
    : public MatrixExpression<T, MatrixScalarBinaryOp<T, E, Op>> {
 public:
    //! Constructs a binary expression for given matrix and scalar.
    MatrixScalarBinaryOp(const E& u, const T& v);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    const E& _u;
    T _v;
    Op _op;
};

//!
//! \brief Vector expression for matrix-vector multiplication.
//!
//! This vector expression represents a matrix-vector operation that takes
//! one input matrix expression and one vector expression.
//!
//! \tparam T   Element value type.
//! \tparam ME  Matrix expression.
//! \tparam VE  Vector expression.
//!
template <typename T, typename ME, typename VE>
class MatrixVectorMul : public VectorExpression<T, MatrixVectorMul<T, ME, VE>> {
 public:
    MatrixVectorMul(const ME& m, const VE& v);

    //! Size of the vector.
    size_t size() const;

    //! Returns vector element at i.
    T operator[](size_t i) const;

 private:
    const ME& _m;
    const VE& _v;
};

//!
//! \brief Matrix expression for matrix-matrix multiplication.
//!
//! This matrix expression represents a matrix-matrix operation that takes
//! two input matrices.
//!
//! \tparam T   Element value type.
//! \tparam ME  Matrix expression.
//! \tparam VE  Vector expression.
//!
template <typename T, typename E1, typename E2>
class MatrixMul : public MatrixExpression<T, MatrixMul<T, E1, E2>> {
 public:
    //! Constructs matrix-matrix multiplication expression for given two input
    //! matrices.
    MatrixMul(const E1& u, const E2& v);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    const E1& _u;
    const E2& _v;
};

// MARK: MatrixBinaryOp Aliases

//! Matrix-matrix addition expression.
template <typename T, typename E1, typename E2>
using MatrixAdd = MatrixBinaryOp<T, E1, E2, std::plus<T>>;

//! Matrix-scalar addition expression.
template <typename T, typename E>
using MatrixScalarAdd = MatrixScalarBinaryOp<T, E, std::plus<T>>;

//! Matrix-matrix subtraction expression.
template <typename T, typename E1, typename E2>
using MatrixSub = MatrixBinaryOp<T, E1, E2, std::minus<T>>;

//! Matrix-scalar subtraction expression.
template <typename T, typename E>
using MatrixScalarSub = MatrixScalarBinaryOp<T, E, std::minus<T>>;

//! Matrix-matrix subtraction expression with inversed order.
template <typename T, typename E>
using MatrixScalarRSub = MatrixScalarBinaryOp<T, E, RMinus<T>>;

//! Matrix-scalar multiplication expression.
template <typename T, typename E>
using MatrixScalarMul = MatrixScalarBinaryOp<T, E, std::multiplies<T>>;

//! Matrix-scalar division expression.
template <typename T, typename E>
using MatrixScalarDiv = MatrixScalarBinaryOp<T, E, std::divides<T>>;

//! Matrix-scalar division expression with inversed order.
template <typename T, typename E>
using MatrixScalarRDiv = MatrixScalarBinaryOp<T, E, RDivides<T>>;

// MARK: Operator overloadings

//! Returns a matrix with opposite sign.
template <typename T, typename E>
MatrixScalarMul<T, E> operator-(const MatrixExpression<T, E>& a);

//! Returns a + b (element-size).
template <typename T, typename E1, typename E2>
MatrixAdd<T, E1, E2> operator+(const MatrixExpression<T, E1>& a,
                               const MatrixExpression<T, E2>& b);

//! Returns a + b', where every element of matrix b' is b.
template <typename T, typename E>
MatrixScalarAdd<T, E> operator+(const MatrixExpression<T, E>& a, T b);

//! Returns a' + b, where every element of matrix a' is a.
template <typename T, typename E>
MatrixScalarAdd<T, E> operator+(T a, const MatrixExpression<T, E>& b);

//! Returns a - b (element-size).
template <typename T, typename E1, typename E2>
MatrixSub<T, E1, E2> operator-(const MatrixExpression<T, E1>& a,
                               const MatrixExpression<T, E2>& b);

//! Returns a - b', where every element of matrix b' is b.
template <typename T, typename E>
MatrixScalarSub<T, E> operator-(const MatrixExpression<T, E>& a, T b);

//! Returns a' - b, where every element of matrix a' is a.
template <typename T, typename E>
MatrixScalarRSub<T, E> operator-(T a, const MatrixExpression<T, E>& b);

//! Returns a * b', where every element of matrix b' is b.
template <typename T, typename E>
MatrixScalarMul<T, E> operator*(const MatrixExpression<T, E>& a, T b);

//! Returns a' * b, where every element of matrix a' is a.
template <typename T, typename E>
MatrixScalarMul<T, E> operator*(T a, const MatrixExpression<T, E>& b);

//! Returns a * b.
template <typename T, typename ME, typename VE>
MatrixVectorMul<T, ME, VE> operator*(const MatrixExpression<T, ME>& a,
                                     const VectorExpression<T, VE>& b);

//! Returns a * b.
template <typename T, typename E1, typename E2>
MatrixMul<T, E1, E2> operator*(const MatrixExpression<T, E1>& a,
                               const MatrixExpression<T, E2>& b);

//! Returns a' / b, where every element of matrix a' is a.
template <typename T, typename E>
MatrixScalarDiv<T, E> operator/(const MatrixExpression<T, E>& a, T b);

//! Returns a / b', where every element of matrix b' is b.
template <typename T, typename E>
MatrixScalarRDiv<T, E> operator/(T a, const MatrixExpression<T, E>& b);

}  // namespace jet

#include "detail/matrix_expression-inl.h"

#endif  // INCLUDE_JET_MATRIX_EXPRESSION_H_
