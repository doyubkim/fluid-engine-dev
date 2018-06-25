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
template <typename E>
Matrix<T, Rows, Cols>::Matrix(const MatrixExpression<T, E>& expression) {
    JET_ASSERT(expression.rows() == Rows && expression.cols() == Cols);

    copyFrom(expression);
}

template <typename T, size_t Rows, size_t Cols>
Matrix<T, Rows, Cols>::Matrix(NestedInitializerListsT<T, 2> lst) {
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
constexpr typename Matrix<T, Rows, Cols>::iterator
Matrix<T, Rows, Cols>::begin() {
    return &_elements[0];
}

template <typename T, size_t Rows, size_t Cols>
constexpr typename Matrix<T, Rows, Cols>::const_iterator
Matrix<T, Rows, Cols>::begin() const {
    return &_elements[0];
}

template <typename T, size_t Rows, size_t Cols>
constexpr typename Matrix<T, Rows, Cols>::iterator
Matrix<T, Rows, Cols>::end() {
    return begin() + Rows * Cols;
}

template <typename T, size_t Rows, size_t Cols>
constexpr typename Matrix<T, Rows, Cols>::const_iterator
Matrix<T, Rows, Cols>::end() const {
    return begin() + Rows * Cols;
}

template <typename T, size_t Rows, size_t Cols>
constexpr typename Matrix<T, Rows, Cols>::pointer
Matrix<T, Rows, Cols>::data() {
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
template <typename E>
Matrix<T, 1, 1>::Matrix(const MatrixExpression<T, E>& expression) {
    JET_ASSERT(expression.rows() == 1 && expression.cols() == 1);

    x = expression.eval(0, 0);
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
constexpr typename Matrix<T, 1, 1>::iterator Matrix<T, 1, 1>::begin() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 1, 1>::const_iterator Matrix<T, 1, 1>::begin()
    const {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 1, 1>::iterator Matrix<T, 1, 1>::end() {
    return begin() + 1;
}

template <typename T>
constexpr typename Matrix<T, 1, 1>::const_iterator Matrix<T, 1, 1>::end()
    const {
    return begin() + 1;
}

template <typename T>
constexpr typename Matrix<T, 1, 1>::pointer Matrix<T, 1, 1>::data() {
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

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix<T, 2, 1> (aka Vector2)

template <typename T>
template <typename E>
Matrix<T, 2, 1>::Matrix(const MatrixExpression<T, E>& expression) {
    JET_ASSERT(expression.rows() == 2 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
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
constexpr typename Matrix<T, 2, 1>::iterator Matrix<T, 2, 1>::begin() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 2, 1>::const_iterator Matrix<T, 2, 1>::begin()
    const {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 2, 1>::iterator Matrix<T, 2, 1>::end() {
    return begin() + 2;
}

template <typename T>
constexpr typename Matrix<T, 2, 1>::const_iterator Matrix<T, 2, 1>::end()
    const {
    return begin() + 2;
}

template <typename T>
constexpr typename Matrix<T, 2, 1>::pointer Matrix<T, 2, 1>::data() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 2, 1>::const_pointer Matrix<T, 2, 1>::data()
    const {
    return &x;
}

template <typename T>
template <typename E>
constexpr T Matrix<T, 2, 1>::cross(
    const MatrixExpression<T, E>& expression) const {
    return x * expression.eval(1, 0) - expression.eval(0, 0) * y;
}

template <typename T>
constexpr Matrix<T, 2, 1> Matrix<T, 2, 1>::reflected(
    const Matrix& normal) const {
    // this - 2(this.n)n
    return (*this) - 2 * this->dot(normal) * normal;
};

template <typename T>
constexpr Matrix<T, 2, 1> Matrix<T, 2, 1>::projected(
    const Matrix& normal) const {
    // this - this.n n
    return (*this) - this->dot(normal) * normal;
}

template <typename T>
constexpr Matrix<T, 2, 1> Matrix<T, 2, 1>::tangential() const {
    return Matrix{-y, x};
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

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix<T, 3, 1> (aka Vector3)

template <typename T>
template <typename E>
Matrix<T, 3, 1>::Matrix(const MatrixExpression<T, E>& expression) {
    JET_ASSERT(expression.rows() == 3 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
    z = expression.eval(2, 0);
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
constexpr typename Matrix<T, 3, 1>::iterator Matrix<T, 3, 1>::begin() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 3, 1>::const_iterator Matrix<T, 3, 1>::begin()
    const {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 3, 1>::iterator Matrix<T, 3, 1>::end() {
    return begin() + 3;
}

template <typename T>
constexpr typename Matrix<T, 3, 1>::const_iterator Matrix<T, 3, 1>::end()
    const {
    return begin() + 3;
}

template <typename T>
constexpr typename Matrix<T, 3, 1>::pointer Matrix<T, 3, 1>::data() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 3, 1>::const_pointer Matrix<T, 3, 1>::data()
    const {
    return &x;
}

template <typename T>
template <typename E>
constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::cross(
    const MatrixExpression<T, E>& expression) const {
    return Matrix<T, 3, 1>(
        y * expression.eval(2, 0) - expression.eval(1, 0) * z,
        z * expression.eval(0, 0) - expression.eval(2, 0) * x,
        x * expression.eval(1, 0) - expression.eval(0, 0) * y);
}

template <typename T>
constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::reflected(
    const Matrix& normal) const {
    // this - 2(this.n)n
    return (*this) - 2 * this->dot(normal) * normal;
};

template <typename T>
constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::projected(
    const Matrix& normal) const {
    // this - this.n n
    return (*this) - this->dot(normal) * normal;
}

template <typename T>
std::tuple<Matrix<T, 3, 1>, Matrix<T, 3, 1>> Matrix<T, 3, 1>::tangential()
    const {
    using V = Matrix<T, 3, 1>;
    V a = ((std::fabs(y) > 0 || std::fabs(z) > 0) ? V(1, 0, 0) : V(0, 1, 0))
              .cross(*this)
              .normalized();
    V b = cross(a);
    return std::make_tuple(a, b);
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

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix<T, 4, 1> (aka Vector4)

template <typename T>
template <typename E>
Matrix<T, 4, 1>::Matrix(const MatrixExpression<T, E>& expression) {
    JET_ASSERT(expression.rows() == 4 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
    z = expression.eval(2, 0);
    w = expression.eval(3, 0);
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
constexpr typename Matrix<T, 4, 1>::iterator Matrix<T, 4, 1>::begin() {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 4, 1>::const_iterator Matrix<T, 4, 1>::begin()
    const {
    return &x;
}

template <typename T>
constexpr typename Matrix<T, 4, 1>::iterator Matrix<T, 4, 1>::end() {
    return begin() + 4;
}

template <typename T>
constexpr typename Matrix<T, 4, 1>::const_iterator Matrix<T, 4, 1>::end()
    const {
    return begin() + 4;
}

template <typename T>
constexpr typename Matrix<T, 4, 1>::pointer Matrix<T, 4, 1>::data() {
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

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix Operators

// MARK: Binary Operators

// *

template <typename T, size_t Rows>
[[deprecated("Use elemMul instead")]] constexpr auto operator*(
    const Vector<T, Rows>& a, const Vector<T, Rows>& b) {
    return MatrixElemWiseMul<T, const Vector<T, Rows>&, const Vector<T, Rows>&>{
        a, b};
}

// /

template <typename T, size_t Rows>
[[deprecated("Use elemDiv instead")]] constexpr auto operator/(
    const Vector<T, Rows>& a, const Vector<T, Rows>& b) {
    return MatrixElemWiseDiv<T, const Vector<T, Rows>&, const Vector<T, Rows>&>{
        a, b()};
}

// MARK: Assignment Operators

// +=

template <typename T, size_t Rows, size_t Cols, typename M2>
void operator+=(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b) {
    a = a + b;
}

template <typename T, size_t Rows, size_t Cols>
void operator+=(Matrix<T, Rows, Cols>& a, const T& b) {
    a = a + b;
}

// -=

template <typename T, size_t Rows, size_t Cols, typename M2>
void operator-=(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b) {
    a = a - b;
}

template <typename T, size_t Rows, size_t Cols>
void operator-=(Matrix<T, Rows, Cols>& a, const T& b) {
    a = a - b;
}

// *=

template <typename T, size_t Rows, size_t Cols, typename M2>
void operator*=(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b) {
    Matrix<T, Rows, Cols> c = a * b;
    a = c;
}

template <typename T, size_t Rows, typename M2>
[[deprecated("Use elemIMul instead")]] void operator*=(
    Matrix<T, Rows, 1>& a, const MatrixExpression<T, M2>& b) {
    a = MatrixElemWiseMul<T, const Matrix<T, Rows, 1>&, const M2&>{a, b()};
}

template <typename T, size_t Rows, size_t Cols, typename M2>
void elemIMul(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b) {
    a = MatrixElemWiseMul<T, const Matrix<T, Rows, Cols>&, const M2&>{a, b()};
}

template <typename T, size_t Rows, size_t Cols>
void operator*=(Matrix<T, Rows, Cols>& a, const T& b) {
    a = MatrixScalarElemWiseMul<T, const Matrix<T, Rows, Cols>&>{a, b};
}

// /=

template <typename T, size_t Rows, size_t Cols, typename M2>
[[deprecated("Use elemIDiv instead")]] void operator/=(
    Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b) {
    a = MatrixElemWiseDiv<T, const Matrix<T, Rows, Cols>&, const M2&>(a, b());
}

template <typename T, size_t Rows, size_t Cols, typename M2>
void elemIDiv(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b) {
    a = MatrixElemWiseDiv<T, const Matrix<T, Rows, Cols>&, const M2&>(a, b());
}

template <typename T, size_t Rows, size_t Cols>
void operator/=(Matrix<T, Rows, Cols>& a, const T& b) {
    a = MatrixScalarElemWiseDiv<T, const Matrix<T, Rows, Cols>&>{a, b};
}

// MARK: Comparison Operators

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), bool> operator==(
    const Matrix<T, Rows, Cols>& a, const Matrix<T, Rows, Cols>& b) {
    return internal::FoldWithAnd<T, Rows, Cols, std::equal_to<T>,
                                 Rows * Cols - 1>::call(a, b,
                                                        std::equal_to<T>());
}

template <typename T, size_t Rows, size_t Cols, typename E>
bool operator==(const Matrix<T, Rows, Cols>& a,
                const MatrixExpression<T, E>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }

    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            if (a(i, j) != b.eval(i, j)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T, size_t Rows, size_t Cols, typename E>
bool operator!=(const Matrix<T, Rows, Cols>& a,
                const MatrixExpression<T, E>& b) {
    return !(a == b);
}

// MARK: Simple Utilities

// Static Accumulate

template <typename T, size_t Rows, size_t Cols, typename BinaryOperation>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a, const T& init, BinaryOperation op) {
    return internal::Reduce<T, Rows, Cols, BinaryOperation, NoOp<T>,
                            Rows * Cols - 1>::call(a, init, op, NoOp<T>());
}

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a, const T& init) {
    return internal::Reduce<T, Rows, Cols, std::plus<T>, NoOp<T>,
                            Rows * Cols - 1>::call(a, init, std::plus<T>(),
                                                   NoOp<T>());
}

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a) {
    return internal::Reduce<T, Rows, Cols, std::plus<T>, NoOp<T>,
                            Rows * Cols - 1>::call(a, std::plus<T>(),
                                                   NoOp<T>());
}

// Dynamic Accumulate

template <typename T, size_t Rows, size_t Cols, typename BinaryOperation>
constexpr std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a, const T& init, BinaryOperation op) {
    return std::accumulate(a.begin(), a.end(), init, op);
}

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a, const T& init) {
    return std::accumulate(a.begin(), a.end(), init, std::plus<T>());
}

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a) {
    return std::accumulate(a.begin(), a.end(), T{}, std::plus<T>());
}

// Product

template <typename T, size_t Rows, size_t Cols>
constexpr T product(const Matrix<T, Rows, Cols>& a, const T& init) {
    return accumulate(a, init, std::multiplies<T>());
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_STATIC_MATRIX_INL_H_
