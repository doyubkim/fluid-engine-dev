// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_OPERATION_H_
#define INCLUDE_JET_MATRIX_OPERATION_H_

#include <jet/functors.h>
#include <jet/nested_initializer_list.h>

#include <functional>

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

// Derived type should be constructible.
template <typename T, size_t Rows, size_t Cols, typename Derived>
class MatrixOperation {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    // MARK: Simple setters/modifiers

    //! Copies from generic expression.
    template <typename E>
    void copyFrom(const MatrixExpression<T, E>& expression);

    //! Sets diagonal elements with input scalar.
    void setDiagonal(const_reference val);

    //! Sets off-diagonal elements with input scalar.
    void setOffDiagonal(const_reference val);

    //! Sets i-th row with input column vector.
    template <typename E>
    void setRow(size_t i, const MatrixExpression<T, E>& row);

    //! Sets i-th column with input vector.
    template <typename E>
    void setColumn(size_t i, const MatrixExpression<T, E>& col);

    void normalize();

    //! Transposes this matrix.
    void transpose();

    //! Inverts this matrix.
    void invert();

    // MARK: Simple getters

    //! Returns true if this matrix is similar to the input matrix within the
    //! given tolerance.
    template <typename E>
    constexpr bool isSimilar(
        const MatrixExpression<T, E>& m,
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

    Derived normalized() const;

    //! Returns diagonal part of this matrix.
    MatrixDiagonal<T, const Derived&> diagonal() const;

    //! Returns off-diagonal part of this matrix.
    MatrixOffDiagonal<T, const Derived&> offDiagonal() const;

    //! Returns strictly lower triangle part of this matrix.
    MatrixTri<T, const Derived&> strictLowerTri() const;

    //! Returns strictly upper triangle part of this matrix.
    MatrixTri<T, const Derived&> strictUpperTri() const;

    //! Returns lower triangle part of this matrix (including the diagonal).
    MatrixTri<T, const Derived&> lowerTri() const;

    //! Returns upper triangle part of this matrix (including the diagonal).
    MatrixTri<T, const Derived&> upperTri() const;

    MatrixTranspose<T, const Derived&> transposed() const;

    //! Returns inverse matrix.
    Derived inverse() const;

    template <typename U>
    MatrixTypeCast<T, U, const Derived&> castTo() const;

    //! Returns the distance to the other vector.
    template <typename E>
    value_type distanceTo(const MatrixExpression<T, E>& other) const;

    //! Returns the squared distance to the other vector.
    template <typename E>
    value_type distanceSquaredTo(const MatrixExpression<T, E>& other) const;

    // MARK: Binary Operators

    template <typename E, typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || Cols == 1), U> dot(
        const MatrixExpression<T, E>& expression) const;

    // MARK: Operator Overloadings

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;

    reference operator()(size_t i, size_t j);

    const_reference operator()(size_t i, size_t j) const;

    //! Copies from generic expression
    template <typename E>
    MatrixOperation& operator=(const MatrixExpression<T, E>& expression);

    // MARK: Builders

    //! Makes a static matrix with zero entries.
    template <typename U = value_type>
    static std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), MatrixConstant<U>>
    makeZero();

    //! Makes a dynamic matrix with zero entries.
    template <typename U = value_type>
    static std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(),
                            MatrixConstant<U>>
    makeZero(size_t rows, size_t cols);

    //! Makes a static identity matrix.
    template <typename U = value_type>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(),
                            MatrixDiagonal<U, MatrixConstant<U>>>
    makeIdentity();

    //! Makes a dynamic identity matrix.
    template <typename U = value_type>
    static std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(),
                            MatrixDiagonal<U, MatrixConstant<U>>>
    makeIdentity(size_t rows);

    //! Makes scale matrix.
    template <typename... Args, typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
    makeScaleMatrix(value_type first, Args... rest);

    //! Makes scale matrix.
    template <typename E, typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
    makeScaleMatrix(const MatrixExpression<T, E>& expression);

    //! Makes rotation matrix.
    //! \warning Input angle should be radian.
    template <typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 2),
                            D>
    makeRotationMatrix(T rad);

    //! Makes rotation matrix.
    //! \warning Input angle should be radian.
    template <typename E, typename D = Derived>
    static std::enable_if_t<
        isMatrixStaticSquare<Rows, Cols>() && (Rows == 3 || Rows == 4), D>
    makeRotationMatrix(const MatrixExpression<T, E>& axis, T rad);

    //! Makes translation matrix.
    template <typename E, typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 4),
                            D>
    makeTranslationMatrix(const MatrixExpression<T, E>& t);

 protected:
    MatrixOperation() = default;

 private:
    // MARK: Private Helpers

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr auto begin();

    constexpr auto begin() const;

    constexpr auto end();

    constexpr auto end() const;

    Derived& operator()();

    const Derived& operator()() const;

    constexpr static T determinant(const MatrixOperation<T, 1, 1, Derived>& m);

    constexpr static T determinant(const MatrixOperation<T, 2, 2, Derived>& m);

    constexpr static T determinant(const MatrixOperation<T, 3, 3, Derived>& m);

    constexpr static T determinant(const MatrixOperation<T, 4, 4, Derived>& m);

    template <typename U = value_type>
    static std::enable_if_t<
        (Rows > 4 && Cols > 4) || isMatrixSizeDynamic<Rows, Cols>(), U>
    determinant(const MatrixOperation<T, Rows, Cols, Derived>& m);

    static void inverse(const MatrixOperation<T, 1, 1, Derived>& m,
                        Derived& result);

    static void inverse(const MatrixOperation<T, 2, 2, Derived>& m,
                        Derived& result);

    static void inverse(const MatrixOperation<T, 3, 3, Derived>& m,
                        Derived& result);

    static void inverse(const MatrixOperation<T, 4, 4, Derived>& m,
                        Derived& result);

    template <typename D = Derived>
    static void inverse(const MatrixOperation& m,
                        std::enable_if_t<(Rows > 4 && Cols > 4) ||
                                             isMatrixSizeDynamic<Rows, Cols>(),
                                         D>& result);
};

}  // namespace jet

#include <jet/detail/matrix_operation-inl.h>

#endif  // INCLUDE_JET_MATRIX_OPERATION_H_
