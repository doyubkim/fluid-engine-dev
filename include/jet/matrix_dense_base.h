// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_DENSE_BASE_H_
#define INCLUDE_JET_MATRIX_DENSE_BASE_H_

#include <jet/functors.h>
#include <jet/matrix_expression.h>
#include <jet/nested_initializer_list.h>

#include <functional>

namespace jet {

// Derived type should be constructible.
template <typename T, size_t Rows, size_t Cols, typename Derived>
class MatrixDenseBase {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    // MARK: Simple setters/modifiers

    //! Copies from generic expression.
    template <size_t R, size_t C, typename E>
    void copyFrom(const MatrixExpression<T, R, C, E>& expression);

    //! Sets diagonal elements with input scalar.
    void setDiagonal(const_reference val);

    //! Sets off-diagonal elements with input scalar.
    void setOffDiagonal(const_reference val);

    //! Sets i-th row with input column vector.
    template <size_t R, size_t C, typename E>
    void setRow(size_t i, const MatrixExpression<T, R, C, E>& row);

    //! Sets i-th column with input vector.
    template <size_t R, size_t C, typename E>
    void setColumn(size_t i, const MatrixExpression<T, R, C, E>& col);

    void normalize();

    //! Transposes this matrix.
    void transpose();

    //! Inverts this matrix.
    void invert();

    // MARK: Operator Overloadings

    reference operator()(size_t i, size_t j);

    const_reference operator()(size_t i, size_t j) const;

    //! Copies from generic expression
    template <size_t R, size_t C, typename E>
    MatrixDenseBase& operator=(const MatrixExpression<T, R, C, E>& expression);

    // MARK: Builders

    //! Makes a static matrix with zero entries.
    template <typename D = Derived>
    static std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), D> makeZero();

    //! Makes a dynamic matrix with zero entries.
    template <typename D = Derived>
    static std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), D> makeZero(
        size_t rows, size_t cols);

    //! Makes a static identity matrix.
    template <typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
    makeIdentity();

    //! Makes a dynamic identity matrix.
    template <typename D = Derived>
    static std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), D> makeIdentity(
        size_t rows);

    //! Makes scale matrix.
    template <typename... Args, typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
    makeScaleMatrix(value_type first, Args... rest);

    //! Makes scale matrix.
    template <size_t R, size_t C, typename E, typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
    makeScaleMatrix(const MatrixExpression<T, R, C, E>& expression);

    //! Makes rotation matrix.
    //! \warning Input angle should be radian.
    template <typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 2),
                            D>
    makeRotationMatrix(T rad);

    //! Makes rotation matrix.
    //! \warning Input angle should be radian.
    template <size_t R, size_t C, typename E, typename D = Derived>
    static std::enable_if_t<
        isMatrixStaticSquare<Rows, Cols>() && (Rows == 3 || Rows == 4), D>
    makeRotationMatrix(const MatrixExpression<T, R, C, E>& axis, T rad);

    //! Makes translation matrix.
    template <size_t R, size_t C, typename E, typename D = Derived>
    static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 4),
                            D>
    makeTranslationMatrix(const MatrixExpression<T, R, C, E>& t);

 protected:
    MatrixDenseBase() = default;

 private:
    // MARK: Private Helpers

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr auto begin();

    constexpr auto begin() const;

    constexpr auto end();

    constexpr auto end() const;

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;

    Derived& operator()();

    const Derived& operator()() const;
};

}  // namespace jet

#include <jet/detail/matrix_dense_base-inl.h>

#endif  // INCLUDE_JET_MATRIX_DENSE_BASE_H_
