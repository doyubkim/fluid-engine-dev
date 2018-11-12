// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_STATIC_MATRIX_INL_H_
#define INCLUDE_JET_DETAIL_STATIC_MATRIX_INL_H_

#include <jet/matrix.h>

#include <cmath>

namespace jet {

// MARK: Internal Helpers

namespace internal {

// TODO: With C++17, fold expression could be used instead.
template <typename M1, typename M2, size_t J>
struct DotProduct {
    constexpr static auto call(const M1& a, const M2& b, size_t i, size_t j) {
        return DotProduct<M1, M2, J - 1>::call(a, b, i, j) + a(i, J) * b(J, j);
    }
};

template <typename M1, typename M2>
struct DotProduct<M1, M2, 0> {
    constexpr static auto call(const M1& a, const M2& b, size_t i, size_t j) {
        return a(i, 0) * b(0, j);
    }
};

// TODO: With C++17, fold expression could be used instead.
template <typename T, size_t Rows, size_t Cols, typename ReduceOperation,
          typename UnaryOperation, size_t I>
struct Reduce {
    // For vector-like Matrix
    template <typename U = T>
    constexpr static std::enable_if_t<(Cols == 1), U> call(
        const Matrix<T, Rows, 1>& a, const T& init, ReduceOperation op,
        UnaryOperation uop) {
        return op(
            Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, I - 1>::call(
                a, init, op, uop),
            uop(a(I, 0)));
    }

    // For vector-like Matrix with zero init
    template <typename U = T>
    constexpr static std::enable_if_t<(Cols == 1), U> call(
        const Matrix<T, Rows, 1>& a, ReduceOperation op, UnaryOperation uop) {
        return op(
            Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, I - 1>::call(
                a, op, uop),
            uop(a(I, 0)));
    }

    // For Matrix
    constexpr static T call(const Matrix<T, Rows, Cols>& a, const T& init,
                            ReduceOperation op, UnaryOperation uop) {
        return op(
            Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, I - 1>::call(
                a, init, op, uop),
            uop(a[I]));
    }

    // For Matrix with zero init
    constexpr static T call(const Matrix<T, Rows, Cols>& a, ReduceOperation op,
                            UnaryOperation uop) {
        return op(
            Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, I - 1>::call(
                a, op, uop),
            uop(a[I]));
    }

    // For diagonal elements on Matrix
    constexpr static T callDiag(const Matrix<T, Rows, Cols>& a, const T& init,
                                ReduceOperation op, UnaryOperation uop) {
        return op(Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation,
                         I - 1>::callDiag(a, init, op, uop),
                  uop(a(I, I)));
    }
};

template <typename T, size_t Rows, size_t Cols, typename ReduceOperation,
          typename UnaryOperation>
struct Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, 0> {
    // For vector-like Matrix
    template <typename U = T>
    constexpr static std::enable_if_t<(Cols > 1), U> call(
        const Matrix<T, Rows, 1>& a, const T& init, ReduceOperation op,
        UnaryOperation uop) {
        return op(uop(a(0, 0)), init);
    }

    // For vector-like Matrix with zero init
    template <typename U = T>
    constexpr static std::enable_if_t<(Cols == 1), U> call(
        const Matrix<T, Rows, 1>& a, ReduceOperation op, UnaryOperation uop) {
        return uop(a(0, 0));
    }

    // For Matrix
    constexpr static T call(const Matrix<T, Rows, Cols>& a, const T& init,
                            ReduceOperation op, UnaryOperation uop) {
        return op(uop(a[0]), init);
    }

    // For MatrixBase with zero init
    constexpr static T call(const Matrix<T, Rows, Cols>& a, ReduceOperation op,
                            UnaryOperation uop) {
        return uop(a[0]);
    }

    // For diagonal elements on MatrixBase
    constexpr static T callDiag(const Matrix<T, Rows, Cols>& a, const T& init,
                                ReduceOperation op, UnaryOperation uop) {
        return op(uop(a(0, 0)), init);
    }
};

// We can use std::logical_and<>, but explicitly putting && helps compiler
// to early terminate the loop (at least for gcc 8.1 as I checked the
// assembly).
// TODO: With C++17, fold expression could be used instead.
template <typename T, size_t Rows, size_t Cols, typename BinaryOperation,
          size_t I>
struct FoldWithAnd {
    constexpr static bool call(const Matrix<T, Rows, Cols>& a,
                               const Matrix<T, Rows, Cols>& b,
                               BinaryOperation op) {
        return FoldWithAnd<T, Rows, Cols, BinaryOperation, I - 1>::call(a, b,
                                                                        op) &&
               op(a[I], b[I]);
    }
};

template <typename T, size_t Rows, size_t Cols, typename BinaryOperation>
struct FoldWithAnd<T, Rows, Cols, BinaryOperation, 0> {
    constexpr static bool call(const Matrix<T, Rows, Cols>& a,
                               const Matrix<T, Rows, Cols>& b,
                               BinaryOperation op) {
        return op(a[0], b[0]);
    }
};

}  // namespace internal

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix Class (Static)

template <typename T, size_t Rows, size_t Cols>
Matrix<T, Rows, Cols>::Matrix(const_reference value) {
    fill(value);
}

template <typename T, size_t Rows, size_t Cols>
template <size_t R, size_t C, typename E>
Matrix<T, Rows, Cols>::Matrix(const MatrixExpression<T, R, C, E>& expression) {
    JET_ASSERT(expression.rows() == Rows && expression.cols() == Cols);

    copyFrom(expression);
}

template <typename T, size_t Rows, size_t Cols>
Matrix<T, Rows, Cols>::Matrix(const NestedInitializerListsT<T, 2>& lst) {
    size_t i = 0;
    for (auto rows : lst) {
        JET_ASSERT(i < Rows);
        size_t j = 0;
        for (auto col : rows) {
            JET_ASSERT(j < Cols);
            (*this)(i, j) = col;
            ++j;
        }
        ++i;
    }
}

template <typename T, size_t Rows, size_t Cols>
Matrix<T, Rows, Cols>::Matrix(const_pointer ptr) {
    size_t cnt = 0;
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j) {
            (*this)(i, j) = ptr[cnt++];
        }
    }
}

template <typename T, size_t Rows, size_t Cols>
void Matrix<T, Rows, Cols>::fill(const T& val) {
    _elements.fill(val);
}

template <typename T, size_t Rows, size_t Cols>
void Matrix<T, Rows, Cols>::fill(const std::function<T(size_t i)>& func) {
    for (size_t i = 0; i < Rows * Cols; ++i) {
        _elements[i] = func(i);
    }
}

template <typename T, size_t Rows, size_t Cols>
void Matrix<T, Rows, Cols>::fill(
    const std::function<T(size_t i, size_t j)>& func) {
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j) {
            (*this)(i, j) = func(i, j);
        }
    }
}

template <typename T, size_t Rows, size_t Cols>
void Matrix<T, Rows, Cols>::swap(Matrix& other) {
    _elements.swap(other._elements);
}

template <typename T, size_t Rows, size_t Cols>
constexpr size_t Matrix<T, Rows, Cols>::rows() const {
    return Rows;
}

template <typename T, size_t Rows, size_t Cols>
constexpr size_t Matrix<T, Rows, Cols>::cols() const {
    return Cols;
}

template <typename T, size_t Rows, size_t Cols>
typename Matrix<T, Rows, Cols>::iterator Matrix<T, Rows, Cols>::begin() {
    return &_elements[0];
}

template <typename T, size_t Rows, size_t Cols>
constexpr typename Matrix<T, Rows, Cols>::const_iterator
Matrix<T, Rows, Cols>::begin() const {
    return &_elements[0];
}

template <typename T, size_t Rows, size_t Cols>
typename Matrix<T, Rows, Cols>::iterator Matrix<T, Rows, Cols>::end() {
    return begin() + Rows * Cols;
}

template <typename T, size_t Rows, size_t Cols>
constexpr typename Matrix<T, Rows, Cols>::const_iterator
Matrix<T, Rows, Cols>::end() const {
    return begin() + Rows * Cols;
}

template <typename T, size_t Rows, size_t Cols>
typename Matrix<T, Rows, Cols>::pointer Matrix<T, Rows, Cols>::data() {
    return &_elements[0];
}

template <typename T, size_t Rows, size_t Cols>
constexpr typename Matrix<T, Rows, Cols>::const_pointer
Matrix<T, Rows, Cols>::data() const {
    return &_elements[0];
}

template <typename T, size_t Rows, size_t Cols>
typename Matrix<T, Rows, Cols>::reference Matrix<T, Rows, Cols>::operator[](
    size_t i) {
    JET_ASSERT(i < Rows * Cols);
    return _elements[i];
}

template <typename T, size_t Rows, size_t Cols>
typename Matrix<T, Rows, Cols>::const_reference Matrix<T, Rows, Cols>::
operator[](size_t i) const {
    JET_ASSERT(i < Rows * Cols);
    return _elements[i];
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix<T, 1, 1> (aka Vector1)

template <typename T>
template <size_t R, size_t C, typename E>
Matrix<T, 1, 1>::Matrix(const MatrixExpression<T, R, C, E>& expression) {
    JET_ASSERT(expression.rows() == 1 && expression.cols() == 1);

    x = expression.eval(0, 0);
}

template <typename T>
Matrix<T, 1, 1>::Matrix(const std::initializer_list<T>& lst) {
    JET_ASSERT(lst.size() > 0);

    x = static_cast<T>(*lst.begin());
}

template <typename T>
void Matrix<T, 1, 1>::fill(const T& val) {
    x = val;
}

template <typename T>
void Matrix<T, 1, 1>::fill(const std::function<T(size_t i)>& func) {
    x = func(0);
}

template <typename T>
void Matrix<T, 1, 1>::fill(const std::function<T(size_t i, size_t j)>& func) {
    x = func(0, 0);
}

template <typename T>
void Matrix<T, 1, 1>::swap(Matrix& other) {
    std::swap(x, other.x);
}

template <typename T>
constexpr size_t Matrix<T, 1, 1>::rows() const {
    return 1;
}

template <typename T>
constexpr size_t Matrix<T, 1, 1>::cols() const {
    return 1;
}

template <typename T>
typename Matrix<T, 1, 1>::iterator Matrix<T, 1, 1>::begin() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 1, 1>::const_iterator Matrix<T, 1, 1>::begin()
    const {
    return &x;
}

template <typename T>
typename Matrix<T, 1, 1>::iterator Matrix<T, 1, 1>::end() {
    return begin() + 1;
}

template <typename T>
constexpr typename Matrix<T, 1, 1>::const_iterator Matrix<T, 1, 1>::end()
    const {
    return begin() + 1;
}

template <typename T>
typename Matrix<T, 1, 1>::pointer Matrix<T, 1, 1>::data() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 1, 1>::const_pointer Matrix<T, 1, 1>::data()
    const {
    return &x;
}

template <typename T>
typename Matrix<T, 1, 1>::reference Matrix<T, 1, 1>::operator[](size_t i) {
    JET_ASSERT(i < 1);
    return (&x)[i];
}

template <typename T>
typename Matrix<T, 1, 1>::const_reference Matrix<T, 1, 1>::operator[](
    size_t i) const {
    JET_ASSERT(i < 1);
    return (&x)[i];
}

template <typename T>
constexpr Matrix<T, 1, 1> Matrix<T, 1, 1>::makeUnitX() {
    return Matrix<T, 1, 1>(1);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix<T, 2, 1> (aka Vector2)

template <typename T>
template <size_t R, size_t C, typename E>
Matrix<T, 2, 1>::Matrix(const MatrixExpression<T, R, C, E>& expression) {
    JET_ASSERT(expression.rows() == 2 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
}

template <typename T>
Matrix<T, 2, 1>::Matrix(const std::initializer_list<T>& lst) {
    JET_ASSERT(lst.size() > 1);

    auto iter = lst.begin();
    x = static_cast<T>(*(iter++));
    y = static_cast<T>(*(iter));
}

template <typename T>
void Matrix<T, 2, 1>::fill(const T& val) {
    x = y = val;
}

template <typename T>
void Matrix<T, 2, 1>::fill(const std::function<T(size_t i)>& func) {
    x = func(0);
    y = func(1);
}

template <typename T>
void Matrix<T, 2, 1>::fill(const std::function<T(size_t i, size_t j)>& func) {
    x = func(0, 0);
    y = func(1, 0);
}

template <typename T>
void Matrix<T, 2, 1>::swap(Matrix& other) {
    std::swap(x, other.x);
    std::swap(y, other.y);
}

template <typename T>
constexpr size_t Matrix<T, 2, 1>::rows() const {
    return 2;
}

template <typename T>
constexpr size_t Matrix<T, 2, 1>::cols() const {
    return 1;
}

template <typename T>
typename Matrix<T, 2, 1>::iterator Matrix<T, 2, 1>::begin() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 2, 1>::const_iterator Matrix<T, 2, 1>::begin()
    const {
    return &x;
}

template <typename T>
typename Matrix<T, 2, 1>::iterator Matrix<T, 2, 1>::end() {
    return begin() + 2;
}

template <typename T>
constexpr typename Matrix<T, 2, 1>::const_iterator Matrix<T, 2, 1>::end()
    const {
    return begin() + 2;
}

template <typename T>
typename Matrix<T, 2, 1>::pointer Matrix<T, 2, 1>::data() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 2, 1>::const_pointer Matrix<T, 2, 1>::data()
    const {
    return &x;
}

template <typename T>
typename Matrix<T, 2, 1>::reference Matrix<T, 2, 1>::operator[](size_t i) {
    JET_ASSERT(i < 2);
    return (&x)[i];
}

template <typename T>
typename Matrix<T, 2, 1>::const_reference Matrix<T, 2, 1>::operator[](
    size_t i) const {
    JET_ASSERT(i < 2);
    return (&x)[i];
}

template <typename T>
constexpr Matrix<T, 2, 1> Matrix<T, 2, 1>::makeUnitX() {
    return Matrix<T, 2, 1>(1, 0);
}

template <typename T>
constexpr Matrix<T, 2, 1> Matrix<T, 2, 1>::makeUnitY() {
    return Matrix<T, 2, 1>(0, 1);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix<T, 3, 1> (aka Vector3)

template <typename T>
template <size_t R, size_t C, typename E>
Matrix<T, 3, 1>::Matrix(const MatrixExpression<T, R, C, E>& expression) {
    JET_ASSERT(expression.rows() == 3 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
    z = expression.eval(2, 0);
}

template <typename T>
Matrix<T, 3, 1>::Matrix(const std::initializer_list<T>& lst) {
    JET_ASSERT(lst.size() > 2);

    auto iter = lst.begin();
    x = static_cast<T>(*(iter++));
    y = static_cast<T>(*(iter++));
    z = static_cast<T>(*(iter));
}

template <typename T>
void Matrix<T, 3, 1>::fill(const T& val) {
    x = y = z = val;
}

template <typename T>
void Matrix<T, 3, 1>::fill(const std::function<T(size_t i)>& func) {
    x = func(0);
    y = func(1);
    z = func(2);
}

template <typename T>
void Matrix<T, 3, 1>::fill(const std::function<T(size_t i, size_t j)>& func) {
    x = func(0, 0);
    y = func(1, 0);
    z = func(2, 0);
}

template <typename T>
void Matrix<T, 3, 1>::swap(Matrix& other) {
    std::swap(x, other.x);
    std::swap(y, other.y);
    std::swap(z, other.z);
}

template <typename T>
constexpr size_t Matrix<T, 3, 1>::rows() const {
    return 3;
}

template <typename T>
constexpr size_t Matrix<T, 3, 1>::cols() const {
    return 1;
}

template <typename T>
typename Matrix<T, 3, 1>::iterator Matrix<T, 3, 1>::begin() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 3, 1>::const_iterator Matrix<T, 3, 1>::begin()
    const {
    return &x;
}

template <typename T>
typename Matrix<T, 3, 1>::iterator Matrix<T, 3, 1>::end() {
    return begin() + 3;
}

template <typename T>
constexpr typename Matrix<T, 3, 1>::const_iterator Matrix<T, 3, 1>::end()
    const {
    return begin() + 3;
}

template <typename T>
typename Matrix<T, 3, 1>::pointer Matrix<T, 3, 1>::data() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 3, 1>::const_pointer Matrix<T, 3, 1>::data()
    const {
    return &x;
}

template <typename T>
typename Matrix<T, 3, 1>::reference Matrix<T, 3, 1>::operator[](size_t i) {
    JET_ASSERT(i < 3);
    return (&x)[i];
}

template <typename T>
typename Matrix<T, 3, 1>::const_reference Matrix<T, 3, 1>::operator[](
    size_t i) const {
    JET_ASSERT(i < 3);
    return (&x)[i];
}

template <typename T>
constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::makeUnitX() {
    return Matrix<T, 3, 1>(1, 0, 0);
}

template <typename T>
constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::makeUnitY() {
    return Matrix<T, 3, 1>(0, 1, 0);
}

template <typename T>
constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::makeUnitZ() {
    return Matrix<T, 3, 1>(0, 0, 1);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix<T, 4, 1> (aka Vector4)

template <typename T>
template <size_t R, size_t C, typename E>
Matrix<T, 4, 1>::Matrix(const MatrixExpression<T, R, C, E>& expression) {
    JET_ASSERT(expression.rows() == 4 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
    z = expression.eval(2, 0);
    w = expression.eval(3, 0);
}

template <typename T>
Matrix<T, 4, 1>::Matrix(const std::initializer_list<T>& lst) {
    JET_ASSERT(lst.size() > 3);

    auto iter = lst.begin();
    x = static_cast<T>(*(iter++));
    y = static_cast<T>(*(iter++));
    z = static_cast<T>(*(iter++));
    w = static_cast<T>(*(iter));
}

template <typename T>
void Matrix<T, 4, 1>::fill(const T& val) {
    x = y = z = w = val;
}

template <typename T>
void Matrix<T, 4, 1>::fill(const std::function<T(size_t i)>& func) {
    x = func(0);
    y = func(1);
    z = func(2);
    w = func(3);
}

template <typename T>
void Matrix<T, 4, 1>::fill(const std::function<T(size_t i, size_t j)>& func) {
    x = func(0, 0);
    y = func(1, 0);
    z = func(2, 0);
    w = func(3, 0);
}

template <typename T>
void Matrix<T, 4, 1>::swap(Matrix& other) {
    std::swap(x, other.x);
    std::swap(y, other.y);
    std::swap(z, other.z);
    std::swap(w, other.w);
}

template <typename T>
constexpr size_t Matrix<T, 4, 1>::rows() const {
    return 4;
}

template <typename T>
constexpr size_t Matrix<T, 4, 1>::cols() const {
    return 1;
}

template <typename T>
typename Matrix<T, 4, 1>::iterator Matrix<T, 4, 1>::begin() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 4, 1>::const_iterator Matrix<T, 4, 1>::begin()
    const {
    return &x;
}

template <typename T>
typename Matrix<T, 4, 1>::iterator Matrix<T, 4, 1>::end() {
    return begin() + 4;
}

template <typename T>
constexpr typename Matrix<T, 4, 1>::const_iterator Matrix<T, 4, 1>::end()
    const {
    return begin() + 4;
}

template <typename T>
typename Matrix<T, 4, 1>::pointer Matrix<T, 4, 1>::data() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 4, 1>::const_pointer Matrix<T, 4, 1>::data()
    const {
    return &x;
}

template <typename T>
typename Matrix<T, 4, 1>::reference Matrix<T, 4, 1>::operator[](size_t i) {
    JET_ASSERT(i < 4);
    return (&x)[i];
}

template <typename T>
typename Matrix<T, 4, 1>::const_reference Matrix<T, 4, 1>::operator[](
    size_t i) const {
    JET_ASSERT(i < 4);
    return (&x)[i];
}

template <typename T>
constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnitX() {
    return Matrix<T, 4, 1>(1, 0, 0, 0);
}

template <typename T>
constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnitY() {
    return Matrix<T, 4, 1>(0, 1, 0, 0);
}

template <typename T>
constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnitZ() {
    return Matrix<T, 4, 1>(0, 0, 1, 0);
}

template <typename T>
constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnitW() {
    return Matrix<T, 4, 1>(0, 0, 0, 1);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix Class (Dynamic)

template <typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix() {}

template <typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(
    size_t rows, size_t cols, const_reference value) {
    _elements.resize(rows * cols);
    _rows = rows;
    _cols = cols;
    fill(value);
}

template <typename T>
template <size_t R, size_t C, typename E>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(
    const MatrixExpression<T, R, C, E>& expression)
    : Matrix(expression.rows(), expression.cols()) {
    copyFrom(expression);
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(
    const NestedInitializerListsT<T, 2>& lst) {
    size_t i = 0;
    for (auto rows : lst) {
        size_t j = 0;
        for (auto col : rows) {
            (void)col;
            ++j;
        }
        _cols = j;
        ++i;
    }
    _rows = i;
    _elements.resize(_rows * _cols);

    i = 0;
    for (auto rows : lst) {
        JET_ASSERT(i < _rows);
        size_t j = 0;
        for (auto col : rows) {
            JET_ASSERT(j < _cols);
            (*this)(i, j) = col;
            ++j;
        }
        ++i;
    }
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(size_t rows,
                                                          size_t cols,
                                                          const_pointer ptr)
    : Matrix(rows, cols) {
    size_t cnt = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            (*this)(i, j) = ptr[cnt++];
        }
    }
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(const Matrix& other)
    : _elements(other._elements), _rows(other._rows), _cols(other._cols) {}

template <typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(Matrix&& other) {
    *this = std::move(other);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::fill(const T& val) {
    std::fill(_elements.begin(), _elements.end(), val);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::fill(
    const std::function<T(size_t i)>& func) {
    for (size_t i = 0; i < _elements.size(); ++i) {
        _elements[i] = func(i);
    }
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::fill(
    const std::function<T(size_t i, size_t j)>& func) {
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            (*this)(i, j) = func(i, j);
        }
    }
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::swap(Matrix& other) {
    _elements.swap(other._elements);
    std::swap(_rows, other._rows);
    std::swap(_cols, other._cols);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::resize(
    size_t rows, size_t cols, const_reference val) {
    Matrix newMatrix{rows, cols, val};
    size_t minRows = std::min(rows, _rows);
    size_t minCols = std::min(cols, _cols);
    for (size_t i = 0; i < minRows; ++i) {
        for (size_t j = 0; j < minCols; ++j) {
            newMatrix(i, j) = (*this)(i, j);
        }
    }
    *this = std::move(newMatrix);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::clear() {
    _elements.clear();
    _rows = 0;
    _cols = 0;
}

template <typename T>
size_t Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::rows() const {
    return _rows;
}

template <typename T>
size_t Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::cols() const {
    return _cols;
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::iterator
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::begin() {
    return &_elements[0];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::const_iterator
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::begin() const {
    return &_elements[0];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::iterator
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::end() {
    return begin() + _rows * _cols;
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::const_iterator
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::end() const {
    return begin() + _rows * _cols;
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::pointer
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::data() {
    return &_elements[0];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::const_pointer
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::data() const {
    return &_elements[0];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::reference
    Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::operator[](size_t i) {
    JET_ASSERT(i < _rows * _cols);
    return _elements[i];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::const_reference
    Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::operator[](
        size_t i) const {
    JET_ASSERT(i < _rows * _cols);
    return _elements[i];
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>&
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::operator=(
    const Matrix& other) {
    _elements = other._elements;
    _rows = other._rows;
    _cols = other._cols;
    return *this;
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>&
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::operator=(Matrix&& other) {
    _elements = std::move(other._elements);
    _rows = other._rows;
    _cols = other._cols;
    other._rows = 0;
    other._cols = 0;
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Specialized Matrix for Dynamic Vector Type

template <typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix() {}

template <typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(size_t rows, const_reference value) {
    _elements.resize(rows, value);
}

template <typename T>
template <size_t R, size_t C, typename E>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(
    const MatrixExpression<T, R, C, E>& expression)
    : Matrix(expression.rows(), 1) {
    copyFrom(expression);
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(const std::initializer_list<T>& lst) {
    size_t sz = lst.size();
    _elements.resize(sz);

    size_t i = 0;
    for (auto row : lst) {
        _elements[i] = static_cast<T>(row);
        ++i;
    }
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(size_t rows, const_pointer ptr)
    : Matrix(rows) {
    size_t cnt = 0;
    for (size_t i = 0; i < rows; ++i) {
        (*this)[i] = ptr[cnt++];
    }
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(const Matrix& other)
    : _elements(other._elements) {}

template <typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(Matrix&& other) {
    *this = std::move(other);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::fill(const T& val) {
    std::fill(_elements.begin(), _elements.end(), val);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::fill(
    const std::function<T(size_t i)>& func) {
    for (size_t i = 0; i < _elements.size(); ++i) {
        _elements[i] = func(i);
    }
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::fill(
    const std::function<T(size_t i, size_t j)>& func) {
    for (size_t i = 0; i < rows(); ++i) {
        _elements[i] = func(i, 0);
    }
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::swap(Matrix& other) {
    _elements.swap(other._elements);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::resize(size_t rows,
                                              const_reference val) {
    _elements.resize(rows, val);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::addElement(const_reference newElem) {
    _elements.push_back(newElem);
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::addElement(const Matrix& newElems) {
    _elements.insert(_elements.end(), newElems._elements.begin(),
                     newElems._elements.end());
}

template <typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::clear() {
    _elements.clear();
}

template <typename T>
size_t Matrix<T, kMatrixSizeDynamic, 1>::rows() const {
    return _elements.size();
}

template <typename T>
constexpr size_t Matrix<T, kMatrixSizeDynamic, 1>::cols() const {
    return 1;
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::iterator
Matrix<T, kMatrixSizeDynamic, 1>::begin() {
    return &_elements[0];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::const_iterator
Matrix<T, kMatrixSizeDynamic, 1>::begin() const {
    return &_elements[0];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::iterator
Matrix<T, kMatrixSizeDynamic, 1>::end() {
    return begin() + _elements.size();
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::const_iterator
Matrix<T, kMatrixSizeDynamic, 1>::end() const {
    return begin() + _elements.size();
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::pointer
Matrix<T, kMatrixSizeDynamic, 1>::data() {
    return &_elements[0];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::const_pointer
Matrix<T, kMatrixSizeDynamic, 1>::data() const {
    return &_elements[0];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::reference
    Matrix<T, kMatrixSizeDynamic, 1>::operator[](size_t i) {
    JET_ASSERT(i < _elements.size());
    return _elements[i];
}

template <typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::const_reference
    Matrix<T, kMatrixSizeDynamic, 1>::operator[](size_t i) const {
    JET_ASSERT(i < _elements.size());
    return _elements[i];
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, 1>& Matrix<T, kMatrixSizeDynamic, 1>::operator=(
    const Matrix& other) {
    _elements = other._elements;
    return *this;
}

template <typename T>
Matrix<T, kMatrixSizeDynamic, 1>& Matrix<T, kMatrixSizeDynamic, 1>::operator=(
    Matrix&& other) {
    _elements = std::move(other._elements);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix Operators

// MARK: Binary Operators

// *

template <typename T, size_t Rows>
[[deprecated("Use elemMul instead")]] constexpr auto operator*(
    const Vector<T, Rows>& a, const Vector<T, Rows>& b) {
    return MatrixElemWiseMul<T, Rows, 1, const Vector<T, Rows>&,
                             const Vector<T, Rows>&>{a, b};
}

// /

template <typename T, size_t Rows>
[[deprecated("Use elemDiv instead")]] constexpr auto operator/(
    const Vector<T, Rows>& a, const Vector<T, Rows>& b) {
    return MatrixElemWiseDiv<T, Rows, 1, const Vector<T, Rows>&,
                             const Vector<T, Rows>&>{a, b.derived()};
}

// MARK: Assignment Operators

// +=

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
void operator+=(Matrix<T, R1, C1>& a,
                const MatrixExpression<T, R2, C2, M2>& b) {
    a = a + b;
}

template <typename T, size_t Rows, size_t Cols>
void operator+=(Matrix<T, Rows, Cols>& a, const T& b) {
    a = a + b;
}

// -=

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
void operator-=(Matrix<T, R1, C1>& a,
                const MatrixExpression<T, R2, C2, M2>& b) {
    a = a - b;
}

template <typename T, size_t Rows, size_t Cols>
void operator-=(Matrix<T, Rows, Cols>& a, const T& b) {
    a = a - b;
}

// *=

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
void operator*=(Matrix<T, R1, C1>& a,
                const MatrixExpression<T, R2, C2, M2>& b) {
    JET_ASSERT(a.cols() == b.rows());

    Matrix<T, R1, C2> c = a * b;
    a = c;
}

template <typename T, size_t R1, size_t R2, size_t C2, typename M2>
[[deprecated("Use elemIMul instead")]] void operator*=(
    Matrix<T, R1, 1>& a, const MatrixExpression<T, R2, C2, M2>& b) {
    JET_ASSERT(a.rows() == b.rows() && a.cols() == b.cols());

    a = MatrixElemWiseMul<T, R1, 1, const Matrix<T, R1, 1>&, const M2&>{
        a, b.derived()};
}

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
void elemIMul(Matrix<T, R1, C1>& a, const MatrixExpression<T, R2, C2, M2>& b) {
    JET_ASSERT(a.rows() == b.rows() && a.cols() == b.cols());

    a = MatrixElemWiseMul<T, R1, C1, const Matrix<T, R1, C1>&, const M2&>{
        a, b.derived()};
}

template <typename T, size_t Rows, size_t Cols>
void operator*=(Matrix<T, Rows, Cols>& a, const T& b) {
    a = MatrixScalarElemWiseMul<T, Rows, Cols, const Matrix<T, Rows, Cols>&>{a,
                                                                             b};
}

// /=

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
[[deprecated("Use elemIDiv instead")]] void operator/=(
    Matrix<T, R1, C1>& a, const MatrixExpression<T, R2, C2, M2>& b) {
    a = MatrixElemWiseDiv<T, R1, C1, const Matrix<T, R1, C1>&, const M2&>(
        a, b.derived());
}

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
void elemIDiv(Matrix<T, R1, C1>& a, const MatrixExpression<T, R2, C2, M2>& b) {
    a = MatrixElemWiseDiv<T, R1, C1, const Matrix<T, R1, C1>&, const M2&>(
        a, b.derived());
}

template <typename T, size_t Rows, size_t Cols>
void operator/=(Matrix<T, Rows, Cols>& a, const T& b) {
    a = MatrixScalarElemWiseDiv<T, Rows, Cols, const Matrix<T, Rows, Cols>&>{a,
                                                                             b};
}

// MARK: Comparison Operators

template <typename T, size_t Rows, size_t Cols, typename M1, typename M2>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), bool> operator==(
    const MatrixExpression<T, Rows, Cols, M1>& a,
    const MatrixExpression<T, Rows, Cols, M2>& b) {
    return internal::FoldWithAnd<T, Rows, Cols, std::equal_to<T>,
                                 Rows * Cols - 1>::call(a, b,
                                                        std::equal_to<T>());
}

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M1,
          typename M2>
bool operator==(const MatrixExpression<T, R1, C1, M1>& a,
                const MatrixExpression<T, R2, C2, M2>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }

    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            if (a.eval(i, j) != b.eval(i, j)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M1,
          typename M2>
bool operator!=(const MatrixExpression<T, R1, C1, M1>& a,
                const MatrixExpression<T, R2, C2, M2>& b) {
    return !(a == b);
}

// MARK: Simple Utilities

// Static Accumulate

template <typename T, size_t Rows, size_t Cols, typename M1,
          typename BinaryOperation>
constexpr std::enable_if_t<IsMatrixSizeStatic<Rows, Cols>::value, T> accumulate(
    const MatrixExpression<T, Rows, Cols, M1>& a, const T& init,
    BinaryOperation op) {
    return internal::Reduce<T, Rows, Cols, BinaryOperation, NoOp<T>,
                            Rows * Cols - 1>::call(a, init, op, NoOp<T>());
}

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr std::enable_if_t<IsMatrixSizeStatic<Rows, Cols>::value, T> accumulate(
    const MatrixExpression<T, Rows, Cols, M1>& a, const T& init) {
    return internal::Reduce<T, Rows, Cols, std::plus<T>, NoOp<T>,
                            Rows * Cols - 1>::call(a, init, std::plus<T>(),
                                                   NoOp<T>());
}

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr std::enable_if_t<IsMatrixSizeStatic<Rows, Cols>::value, T> accumulate(
    const MatrixExpression<T, Rows, Cols, M1>& a) {
    return internal::Reduce<T, Rows, Cols, std::plus<T>, NoOp<T>,
                            Rows * Cols - 1>::call(a, std::plus<T>(),
                                                   NoOp<T>());
}

// Dynamic Accumulate

template <typename T, size_t Rows, size_t Cols, typename M1,
          typename BinaryOperation>
constexpr std::enable_if_t<IsMatrixSizeDynamic<Rows, Cols>::value, T>
accumulate(const MatrixExpression<T, Rows, Cols, M1>& a, const T& init,
           BinaryOperation op) {
    return std::accumulate(a.begin(), a.end(), init, op);
}

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr std::enable_if_t<IsMatrixSizeDynamic<Rows, Cols>::value, T>
accumulate(const MatrixExpression<T, Rows, Cols, M1>& a, const T& init) {
    return std::accumulate(a.begin(), a.end(), init, std::plus<T>());
}

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr std::enable_if_t<IsMatrixSizeDynamic<Rows, Cols>::value, T>
accumulate(const MatrixExpression<T, Rows, Cols, M1>& a) {
    return std::accumulate(a.begin(), a.end(), T{}, std::plus<T>());
}

// Product

template <typename T, size_t Rows, size_t Cols, typename M1>
constexpr T product(const MatrixExpression<T, Rows, Cols, M1>& a,
                    const T& init) {
    return accumulate(a, init, std::multiplies<T>());
}

// Interpolation
template <typename T, size_t Rows, size_t Cols, typename M1, typename M2,
          typename M3, typename M4>
std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), Matrix<T, Rows, Cols>>
monotonicCatmullRom(const MatrixExpression<T, Rows, Cols, M1>& f0,
                    const MatrixExpression<T, Rows, Cols, M2>& f1,
                    const MatrixExpression<T, Rows, Cols, M3>& f2,
                    const MatrixExpression<T, Rows, Cols, M4>& f3, T f) {
    Matrix<T, Rows, Cols> result;
    for (size_t i = 0; i < f0.rows(); ++i) {
        for (size_t j = 0; j < f0.cols(); ++j) {
            result(i, j) = monotonicCatmullRom(f0.eval(i, j), f1.eval(i, j),
                                               f2.eval(i, j), f3.eval(i, j), f);
        }
    }

    return result;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_STATIC_MATRIX_INL_H_
