// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_EXPRESSION_H_
#define INCLUDE_JET_MATRIX_EXPRESSION_H_

#include <jet/functors.h>

#include <tuple>

namespace jet {

static constexpr size_t kMatrixSizeDynamic = 0;

template <size_t Rows, size_t Cols>
constexpr bool isMatrixSizeDynamic() {
    return (Rows == kMatrixSizeDynamic) || (Cols == kMatrixSizeDynamic);
}

template <size_t Rows, size_t Cols>
constexpr bool isMatrixSizeStatic() {
    return !isMatrixSizeDynamic<Rows, Cols>();
}

template <size_t Rows, size_t Cols>
constexpr bool isMatrixStaticSquare() {
    return isMatrixSizeStatic<Rows, Cols>() && (Rows == Cols);
}

template <typename T, size_t Rows, size_t Cols>
class Matrix;

template <typename T, size_t Rows, size_t Cols, typename M1>
class MatrixDiagonal;

template <typename T, size_t Rows, size_t Cols, typename M1>
class MatrixOffDiagonal;

template <typename T, size_t Rows, size_t Cols, typename M1>
class MatrixTri;

template <typename T, size_t Rows, size_t Cols, typename M1>
class MatrixTranspose;

template <typename T, size_t Rows, size_t Cols, typename M1,
          typename UnaryOperation>
class MatrixUnaryOp;

template <typename T, size_t Rows, size_t Cols, typename M1,
          typename BinaryOperation>
class MatrixScalarElemWiseBinaryOp;

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixExpression

//!
//! \brief Base class for matrix expression.
//!
//! Matrix expression is a meta type that enables template expression
//! pattern.
//!
//! \tparam T  Real number type.
//! \tparam E  Subclass type.
//!
template <typename T, size_t Rows, size_t Cols, typename Derived>
class MatrixExpression {
 public:
    using value_type = T;

    ////////////////////////////////////////////////////////////////////////////
    // MARK: Core Expression Interface

    //! Returns the number of rows.
    constexpr size_t rows() const;

    //! Returns the number of columns.
    constexpr size_t cols() const;

    //! Returns the evaluated value for (i, j).
    T eval(size_t i, size_t j) const;

    ////////////////////////////////////////////////////////////////////////////
    // MARK: Simple getters

    Matrix<T, Rows, Cols> eval() const;

    //! Returns true if this matrix is similar to the input matrix within the
    //! given tolerance.
    template <size_t R, size_t C, typename E>
    bool isSimilar(const MatrixExpression<T, R, C, E>& m,
                   double tol = std::numeric_limits<double>::epsilon()) const;

    //! Returns true if this matrix is a square matrix.
    constexpr bool isSquare() const;

    constexpr value_type sum() const;

    constexpr value_type avg() const;

    constexpr value_type min() const;

    constexpr value_type max() const;

    value_type absmin() const;

    value_type absmax() const;

    value_type trace() const;

    value_type determinant() const;

    size_t dominantAxis() const;

    size_t subminantAxis() const;

    value_type norm() const;

    value_type normSquared() const;

    value_type frobeniusNorm() const;

    value_type length() const;

    value_type lengthSquared() const;

    //! Returns the distance to the other vector.
    template <size_t R, size_t C, typename E>
    value_type distanceTo(const MatrixExpression<T, R, C, E>& other) const;

    //! Returns the squared distance to the other vector.
    template <size_t R, size_t C, typename E>
    value_type distanceSquaredTo(
        const MatrixExpression<T, R, C, E>& other) const;

    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, const Derived&, std::divides<T>>
    normalized() const;

    //! Returns diagonal part of this matrix.
    MatrixDiagonal<T, Rows, Cols, const Derived&> diagonal() const;

    //! Returns off-diagonal part of this matrix.
    MatrixOffDiagonal<T, Rows, Cols, const Derived&> offDiagonal() const;

    //! Returns strictly lower triangle part of this matrix.
    MatrixTri<T, Rows, Cols, const Derived&> strictLowerTri() const;

    //! Returns strictly upper triangle part of this matrix.
    MatrixTri<T, Rows, Cols, const Derived&> strictUpperTri() const;

    //! Returns lower triangle part of this matrix (including the diagonal).
    MatrixTri<T, Rows, Cols, const Derived&> lowerTri() const;

    //! Returns upper triangle part of this matrix (including the diagonal).
    MatrixTri<T, Rows, Cols, const Derived&> upperTri() const;

    MatrixTranspose<T, Rows, Cols, const Derived&> transposed() const;

    //! Returns inverse matrix.
    Matrix<T, Rows, Cols> inverse() const;

    template <typename U>
    MatrixUnaryOp<U, Rows, Cols, const Derived&, TypeCast<T, U>> castTo() const;

    ////////////////////////////////////////////////////////////////////////////
    // MARK: Binary Operators

    template <size_t R, size_t C, typename E, typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || Cols == 1) &&
                         (isMatrixSizeDynamic<R, C>() || C == 1),
                     U>
    dot(const MatrixExpression<T, R, C, E>& expression) const;

    template <size_t R, size_t C, typename E, typename U = value_type>
    constexpr std::enable_if_t<
        (isMatrixSizeDynamic<Rows, Cols>() || (Rows == 2 && Cols == 1)) &&
            (isMatrixSizeDynamic<R, C>() || (R == 2 && C == 1)),
        U>
    cross(const MatrixExpression<T, R, C, E>& expression) const;

    template <size_t R, size_t C, typename E, typename U = value_type>
    constexpr std::enable_if_t<
        (isMatrixSizeDynamic<Rows, Cols>() || (Rows == 3 && Cols == 1)) &&
            (isMatrixSizeDynamic<R, C>() || (R == 3 && C == 1)),
        Matrix<U, 3, 1>>
    cross(const MatrixExpression<T, R, C, E>& expression) const;

    //! Returns the reflection vector to the surface with given surface normal.
    template <size_t R, size_t C, typename E, typename U = value_type>
    constexpr std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                                ((Rows == 2 || Rows == 3) && Cols == 1)) &&
                                   (isMatrixSizeDynamic<R, C>() ||
                                    ((R == 2 || R == 3) && C == 1)),
                               Matrix<U, Rows, 1>>
    reflected(const MatrixExpression<T, R, C, E>& normal) const;

    //! Returns the projected vector to the surface with given surface normal.
    template <size_t R, size_t C, typename E, typename U = value_type>
    constexpr std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                                ((Rows == 2 || Rows == 3) && Cols == 1)) &&
                                   (isMatrixSizeDynamic<R, C>() ||
                                    ((R == 2 || R == 3) && C == 1)),
                               Matrix<U, Rows, 1>>
    projected(const MatrixExpression<T, R, C, E>& normal) const;

    //! Returns the tangential vector for this vector.
    template <typename U = value_type>
    constexpr std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                                (Rows == 2 && Cols == 1)),
                               Matrix<U, 2, 1>>
    tangential() const;

    //! Returns the tangential vectors for this vector.
    template <typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                      (Rows == 3 && Cols == 1)),
                     std::tuple<Matrix<U, 3, 1>, Matrix<U, 3, 1>>>
    tangentials() const;

    ////////////////////////////////////////////////////////////////////////////
    // MARK: Operator Overloadings

    //! Returns actual implementation (the subclass).
    Derived& operator()();

    //! Returns actual implementation (the subclass).
    const Derived& operator()() const;

 protected:
    // Prohibits constructing this class instance.
    MatrixExpression() = default;

    constexpr static T determinant(const MatrixExpression<T, 1, 1, Derived>& m);

    constexpr static T determinant(const MatrixExpression<T, 2, 2, Derived>& m);

    constexpr static T determinant(const MatrixExpression<T, 3, 3, Derived>& m);

    constexpr static T determinant(const MatrixExpression<T, 4, 4, Derived>& m);

    template <typename U = value_type>
    static std::enable_if_t<
        (Rows > 4 && Cols > 4) || isMatrixSizeDynamic<Rows, Cols>(), U>
    determinant(const MatrixExpression<T, Rows, Cols, Derived>& m);

    static void inverse(const MatrixExpression<T, 1, 1, Derived>& m,
                        Matrix<T, Rows, Cols>& result);

    static void inverse(const MatrixExpression<T, 2, 2, Derived>& m,
                        Matrix<T, Rows, Cols>& result);

    static void inverse(const MatrixExpression<T, 3, 3, Derived>& m,
                        Matrix<T, Rows, Cols>& result);

    static void inverse(const MatrixExpression<T, 4, 4, Derived>& m,
                        Matrix<T, Rows, Cols>& result);

    template <typename M = Matrix<T, Rows, Cols>>
    static void inverse(const MatrixExpression& m,
                        std::enable_if_t<(Rows > 4 && Cols > 4) ||
                                             isMatrixSizeDynamic<Rows, Cols>(),
                                         M>& result);
};

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixConstant

template <typename T, size_t Rows, size_t Cols>
class MatrixConstant
    : public MatrixExpression<T, Rows, Cols, MatrixConstant<T, Rows, Cols>> {
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

template <typename T, size_t Rows, size_t Cols, typename M1>
class MatrixDiagonal
    : public MatrixExpression<T, Rows, Cols,
                              MatrixDiagonal<T, Rows, Cols, M1>> {
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

template <typename T, size_t Rows, size_t Cols, typename M1>
class MatrixOffDiagonal
    : public MatrixExpression<T, Rows, Cols,
                              MatrixOffDiagonal<T, Rows, Cols, M1>> {
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

template <typename T, size_t Rows, size_t Cols, typename M1>
class MatrixTri
    : public MatrixExpression<T, Rows, Cols, MatrixTri<T, Rows, Cols, M1>> {
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

template <typename T, size_t Rows, size_t Cols, typename M1>
class MatrixTranspose
    : public MatrixExpression<T, Rows, Cols,
                              MatrixTranspose<T, Rows, Cols, M1>> {
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

template <typename T, size_t Rows, size_t Cols, typename M1,
          typename UnaryOperation>
class MatrixUnaryOp
    : public MatrixExpression<
          T, Rows, Cols, MatrixUnaryOp<T, Rows, Cols, M1, UnaryOperation>> {
 public:
    constexpr MatrixUnaryOp(const M1& m1) : _m1(m1) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
    UnaryOperation _op;
};

template <typename T, size_t Rows, size_t Cols, typename M1>
using MatrixNegate = MatrixUnaryOp<T, Rows, Cols, M1, std::negate<T>>;

template <typename T, size_t Rows, size_t Cols, typename M1>
using MatrixCeil = MatrixUnaryOp<T, Rows, Cols, M1, Ceil<T>>;

template <typename T, size_t Rows, size_t Cols, typename M1>
using MatrixFloor = MatrixUnaryOp<T, Rows, Cols, M1, Floor<T>>;

template <typename T, size_t Rows, size_t Cols, typename U, typename M1>
using MatrixTypeCast = MatrixUnaryOp<U, Rows, Cols, M1, TypeCast<T, U>>;

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto ceil(const MatrixExpression<T, Rows, Cols, M1>& a);

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto floor(const MatrixExpression<T, Rows, Cols, M1>& a);

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1>& m);

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
template <typename T, size_t Rows, size_t Cols, typename E1, typename E2,
          typename BinaryOperation>
class MatrixElemWiseBinaryOp
    : public MatrixExpression<
          T, Rows, Cols,
          MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, BinaryOperation>> {
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
template <typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseAdd =
    MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, std::plus<T>>;

//! Matrix expression for element-wise matrix-matrix subtraction.
template <typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseSub =
    MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, std::minus<T>>;

//! Matrix expression for element-wise matrix-matrix multiplication.
template <typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseMul =
    MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, std::multiplies<T>>;

//! Matrix expression for element-wise matrix-matrix division.
template <typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseDiv =
    MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, std::divides<T>>;

//! Matrix expression for element-wise matrix-matrix min operation.
template <typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseMin = MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, Min<T>>;

//! Matrix expression for element-wise matrix-matrix max operation.
template <typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseMax = MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, Max<T>>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2>
constexpr auto operator+(const MatrixExpression<T, Rows, Cols, M1>& a,
                         const MatrixExpression<T, Rows, Cols, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2>
constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1>& a,
                         const MatrixExpression<T, Rows, Cols, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2>
constexpr auto elemMul(const MatrixExpression<T, Rows, Cols, M1>& a,
                       const MatrixExpression<T, Rows, Cols, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2>
constexpr auto elemDiv(const MatrixExpression<T, Rows, Cols, M1>& a,
                       const MatrixExpression<T, Rows, Cols, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2>
constexpr auto min(const MatrixExpression<T, Rows, Cols, M1>& a,
                   const MatrixExpression<T, Rows, Cols, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2>
constexpr auto max(const MatrixExpression<T, Rows, Cols, M1>& a,
                   const MatrixExpression<T, Rows, Cols, M2>& b);

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixScalarElemWiseBinaryOp

template <typename T, size_t Rows, size_t Cols, typename M1,
          typename BinaryOperation>
class MatrixScalarElemWiseBinaryOp
    : public MatrixExpression<
          T, Rows, Cols,
          MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, BinaryOperation>> {
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

template <typename T, size_t Rows, size_t Cols, typename M1>
using MatrixScalarElemWiseAdd =
    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, std::plus<T>>;

template <typename T, size_t Rows, size_t Cols, typename M1>
using MatrixScalarElemWiseSub =
    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, std::minus<T>>;

template <typename T, size_t Rows, size_t Cols, typename M1>
using MatrixScalarElemWiseMul =
    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, std::multiplies<T>>;

template <typename T, size_t Rows, size_t Cols, typename M1>
using MatrixScalarElemWiseDiv =
    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, std::divides<T>>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto operator+(const MatrixExpression<T, Rows, Cols, M1>& a,
                         const T& b);

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1>& a,
                         const T& b);

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto operator*(const MatrixExpression<T, Rows, Cols, M1>& a,
                         const T& b);

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr auto operator/(const MatrixExpression<T, Rows, Cols, M1>& a,
                         const T& b);

////////////////////////////////////////////////////////////////////////////////
// MARK: ScalarMatrixElemWiseBinaryOp

template <typename T, size_t Rows, size_t Cols, typename M2,
          typename BinaryOperation>
class ScalarMatrixElemWiseBinaryOp
    : public MatrixExpression<
          T, Rows, Cols,
          ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, BinaryOperation>> {
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

template <typename T, size_t Rows, size_t Cols, typename M2>
using ScalarMatrixElemWiseAdd =
    ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, std::plus<T>>;

template <typename T, size_t Rows, size_t Cols, typename M2>
using ScalarMatrixElemWiseSub =
    ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, std::minus<T>>;

template <typename T, size_t Rows, size_t Cols, typename M2>
using ScalarMatrixElemWiseMul =
    ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, std::multiplies<T>>;

template <typename T, size_t Rows, size_t Cols, typename M2>
using ScalarMatrixElemWiseDiv =
    ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, std::divides<T>>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t Rows, size_t Cols, typename M2>
constexpr auto operator+(const T& a,
                         const MatrixExpression<T, Rows, Cols, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M2>
constexpr auto operator-(const T& a,
                         const MatrixExpression<T, Rows, Cols, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M2>
constexpr auto operator*(const T& a,
                         const MatrixExpression<T, Rows, Cols, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M2>
constexpr auto operator/(const T& a,
                         const MatrixExpression<T, Rows, Cols, M2>& b);

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixTernaryOp

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2,
          typename M3, typename TernaryOperation>
class MatrixTernaryOp
    : public MatrixExpression<
          T, Rows, Cols,
          MatrixTernaryOp<T, Rows, Cols, M1, M2, M3, TernaryOperation>> {
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

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2,
          typename M3>
using MatrixClamp = MatrixTernaryOp<T, Rows, Cols, M1, M2, M3, Clamp<T>>;

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2,
          typename M3>
constexpr auto clamp(const MatrixExpression<T, Rows, Cols, M1>& a,
                     const MatrixExpression<T, Rows, Cols, M2>& low,
                     const MatrixExpression<T, Rows, Cols, M3>& high);

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixMul

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2>
class MatrixMul
    : public MatrixExpression<T, Rows, Cols, MatrixMul<T, Rows, Cols, M1, M2>> {
 public:
    constexpr MatrixMul(const M1& m1, const M2& m2) : _m1(m1), _m2(m2) {}

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    T operator()(size_t i, size_t j) const;

 private:
    M1 _m1;
    M2 _m2;
};

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M1,
          typename M2>
constexpr auto operator*(const MatrixExpression<T, R1, C1, M1>& a,
                         const MatrixExpression<T, R2, C2, M2>& b);

}  // namespace jet

#include <jet/detail/matrix_expression-inl.h>

#endif  // INCLUDE_JET_MATRIX_EXPRESSION_H_
