// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_MATRIX_OPERATION_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX_OPERATION_INL_H_

#include <jet/matrix_operation.h>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: Simple Setters/Modifiers

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename E>
void MatrixOperation<T, Rows, Cols, D>::copyFrom(
    const MatrixExpression<T, E>& expression) {
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            (*this)(i, j) = expression.eval(i, j);
        }
    }
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::setDiagonal(const_reference val) {
    size_t n = std::min(rows(), cols());
    for (size_t i = 0; i < n; ++i) {
        (*this)(i, i) = val;
    }
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::setOffDiagonal(const_reference val) {
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            if (i != j) {
                (*this)(i, j) = val;
            }
        }
    }
}

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename E>
void MatrixOperation<T, Rows, Cols, D>::setRow(
    size_t i, const MatrixExpression<T, E>& row) {
    JET_ASSERT(row.rows() == cols() && row.cols() == 1);
    for (size_t j = 0; j < cols(); ++j) {
        (*this)(i, j) = row.eval(j, 0);
    }
}

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename E>
void MatrixOperation<T, Rows, Cols, D>::setColumn(
    size_t j, const MatrixExpression<T, E>& col) {
    JET_ASSERT(col.rows() == rows() && col.cols() == 1);
    for (size_t i = 0; i < rows(); ++i) {
        (*this)(i, j) = col.eval(i, 0);
    }
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::normalize() {
    (*this)() /= norm();
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::transpose() {
    D tmp = transposed();
    copyFrom(tmp);
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::invert() {
    copyFrom(inverse());
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Simple Getters

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename E>
constexpr bool MatrixOperation<T, Rows, Cols, D>::isSimilar(
    const MatrixExpression<T, E>& expression, double tol) const {
    if (expression.rows() != rows() || expression.cols() != cols()) {
        return false;
    }

    SimilarTo<T> op{tol};
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            if (!op((*this)(i, j), expression.eval(i, j))) {
                return false;
            }
        }
    }
    return true;
}

// template <typename T, size_t Rows, size_t Cols, typename D>
// template <typename E>
// constexpr bool MatrixOperation<T, Rows, Cols, D>::isSimilar(
//    const StaticMatrixExpression<T, E>& m, double tol) const {
//    return internal::FoldWithAnd<T, SimilarTo<T>, Rows * Cols - 1>::call(
//        (*this)(), m(), SimilarTo<T>{tol});
//}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr bool MatrixOperation<T, Rows, Cols, D>::isSquare() const {
    return rows() == cols();
}

// template <typename T, size_t Rows, size_t Cols, typename D>
// constexpr T MatrixOperation<T, Rows, Cols, D>::sum() const {
//    return accumulate(*this);
//}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr T MatrixOperation<T, Rows, Cols, D>::sum() const {
    T s = *begin();
    for (auto iter = begin() + 1; iter != end(); ++iter) {
        s += *iter;
    }
    return s;
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr T MatrixOperation<T, Rows, Cols, D>::avg() const {
    return sum() / (rows() * cols());
}

// template <typename T, size_t Rows, size_t Cols, typename D>
// constexpr T MatrixOperation<T, Rows, Cols, D>::min() const {
//    return internal::Reduce<T, Min<T>, NoOp<T>, Rows * Cols - 2>::call(
//        (*this), (*this)[Rows * Cols - 1], Min<T>{}, NoOp<T>{});
//}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr T MatrixOperation<T, Rows, Cols, D>::min() const {
    return *std::min_element(begin(), end());
}

// template <typename T, size_t Rows, size_t Cols, typename D>
// constexpr T MatrixOperation<T, Rows, Cols, D>::max() const {
//    return internal::Reduce<T, Max<T>, NoOp<T>, Rows * Cols - 2>::call(
//        (*this), (*this)[Rows * Cols - 1], Max<T>{}, NoOp<T>{});
//}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr T MatrixOperation<T, Rows, Cols, D>::max() const {
    return *std::max_element(begin(), end());
}

// template <typename T, size_t Rows, size_t Cols, typename D>
// constexpr T MatrixOperation<T, Rows, Cols, D>::absmin() const {
//    return internal::Reduce<T, AbsMin<T>, NoOp<T>, Rows * Cols - 2>::call(
//        (*this), (*this)[Rows * Cols - 1], AbsMin<T>{}, NoOp<T>{});
//}

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::absmin() const {
    T result = *begin();
    for (auto iter = begin() + 1; iter != end(); ++iter) {
        result = jet::absmin(result, *iter);
    }
    return result;
}

// template <typename T, size_t Rows, size_t Cols, typename D>
// constexpr T MatrixOperation<T, Rows, Cols, D>::absmax() const {
//    return internal::Reduce<T, AbsMax<T>, NoOp<T>, Rows * Cols - 2>::call(
//        (*this), (*this)[Rows * Cols - 1], AbsMax<T>{}, NoOp<T>{});
//}

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::absmax() const {
    T result = *begin();
    for (auto iter = begin() + 1; iter != end(); ++iter) {
        result = jet::absmax(result, *iter);
    }
    return result;
}

// template <typename T, size_t Rows, size_t Cols, typename D>
// template <typename U>
// constexpr std::enable_if_t<(Rows == Cols), U> MatrixOperation<T, Rows, Cols,
// D>::trace()
//    const {
//    return internal::Reduce<T, std::plus<T>, NoOp<T>, Rows - 2>::callDiag(
//        *this, (*this)(Rows - 1, Rows - 1), std::plus<T>{}, NoOp<T>{});
//}

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::trace() const {
    JET_ASSERT(rows() == cols());

    T result = (*this)(0, 0);
    for (size_t i = 1; i < rows(); ++i) {
        result += (*this)(i, i);
    }
    return result;
}

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::determinant() const {
    JET_ASSERT(rows() == cols());

    return determinant(*this);
}

template <typename T, size_t Rows, size_t Cols, typename D>
size_t MatrixOperation<T, Rows, Cols, D>::dominantAxis() const {
    JET_ASSERT(cols() == 1);

    size_t ret = 0;
    T best = (*this)[0];
    for (size_t i = 1; i < rows(); ++i) {
        if (std::fabs((*this)[i]) > std::fabs(best)) {
            best = (*this)[i];
            ret = i;
        }
    }
    return ret;
}

template <typename T, size_t Rows, size_t Cols, typename D>
size_t MatrixOperation<T, Rows, Cols, D>::subminantAxis() const {
    JET_ASSERT(cols() == 1);

    size_t ret = 0;
    T best = (*this)[0];
    for (size_t i = 1; i < rows(); ++i) {
        if (std::fabs((*this)[i]) < std::fabs(best)) {
            best = (*this)[i];
            ret = i;
        }
    }
    return ret;
}

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::norm() const {
    return std::sqrt(normSquared());
}

// template <typename T, size_t Rows, size_t Cols, typename D>
// T MatrixOperation<T, Rows, Cols, D>::normSquared() const {
//    return internal::Reduce<T, std::plus<T>, Square<T>, Rows * Cols -
//    1>::call(
//        (*this), std::plus<T>{}, Square<T>{});
//}

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::normSquared() const {
    T result = (*begin()) * (*begin());
    for (auto iter = begin() + 1; iter != end(); ++iter) {
        result += (*iter) * (*iter);
    }
    return result;
}

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::frobeniusNorm() const {
    return norm();
};

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::length() const {
    JET_ASSERT(cols() == 1);
    return norm();
};

template <typename T, size_t Rows, size_t Cols, typename D>
T MatrixOperation<T, Rows, Cols, D>::lengthSquared() const {
    JET_ASSERT(cols() == 1);
    return normSquared();
};

template <typename T, size_t Rows, size_t Cols, typename D>
D MatrixOperation<T, Rows, Cols, D>::normalized() const {
    return D{(*this)() / norm()};
}

template <typename T, size_t Rows, size_t Cols, typename D>
MatrixDiagonal<T, const D&> MatrixOperation<T, Rows, Cols, D>::diagonal()
    const {
    return MatrixDiagonal<T, const D&>{(*this)()};
}

template <typename T, size_t Rows, size_t Cols, typename D>
MatrixOffDiagonal<T, const D&> MatrixOperation<T, Rows, Cols, D>::offDiagonal()
    const {
    return MatrixOffDiagonal<T, const D&>{(*this)()};
}

template <typename T, size_t Rows, size_t Cols, typename D>
MatrixTri<T, const D&> MatrixOperation<T, Rows, Cols, D>::strictLowerTri()
    const {
    return MatrixTri<T, const D&>{(*this)(), false, true};
}

template <typename T, size_t Rows, size_t Cols, typename D>
MatrixTri<T, const D&> MatrixOperation<T, Rows, Cols, D>::strictUpperTri()
    const {
    return MatrixTri<T, const D&>{(*this)(), true, true};
}

template <typename T, size_t Rows, size_t Cols, typename D>
MatrixTri<T, const D&> MatrixOperation<T, Rows, Cols, D>::lowerTri() const {
    return MatrixTri<T, const D&>{(*this)(), false, false};
}

template <typename T, size_t Rows, size_t Cols, typename D>
MatrixTri<T, const D&> MatrixOperation<T, Rows, Cols, D>::upperTri() const {
    return MatrixTri<T, const D&>{(*this)(), true, false};
}

template <typename T, size_t Rows, size_t Cols, typename D>
MatrixTranspose<T, const D&> MatrixOperation<T, Rows, Cols, D>::transposed()
    const {
    return MatrixTranspose<T, const D&>{(*this)()};
}

template <typename T, size_t Rows, size_t Cols, typename D>
D MatrixOperation<T, Rows, Cols, D>::inverse() const {
    D result;
    inverse(*this, result);
    return result;
}

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename U>
MatrixTypeCast<T, U, const D&> MatrixOperation<T, Rows, Cols, D>::castTo()
    const {
    return MatrixTypeCast<T, U, const D&>{(*this)()};
};

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename E>
T MatrixOperation<T, Rows, Cols, D>::distanceTo(
    const MatrixExpression<T, E>& other) const {
    JET_ASSERT(cols() == 1);
    return std::sqrt(distanceSquaredTo(other));
};

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename E>
T MatrixOperation<T, Rows, Cols, D>::distanceSquaredTo(
    const MatrixExpression<T, E>& other) const {
    JET_ASSERT(cols() == 1);
    return D((*this)() - other()).normSquared();
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Binary Operators

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename E, typename U>
std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || Cols == 1), U>
MatrixOperation<T, Rows, Cols, D>::dot(
    const MatrixExpression<T, E>& expression) const {
    JET_ASSERT(expression.rows() == rows() && expression.cols() == 1);

    T sum = (*this)[0] * expression.eval(0, 0);
    for (size_t i = 1; i < rows(); ++i) {
        sum += (*this)[i] * expression.eval(i, 0);
    }
    return sum;
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Operator Overloadings

template <typename T, size_t Rows, size_t Cols, typename D>
typename MatrixOperation<T, Rows, Cols, D>::reference
    MatrixOperation<T, Rows, Cols, D>::operator[](size_t i) {
    JET_ASSERT(i < rows() * cols());
    return (*this)()[i];
}

template <typename T, size_t Rows, size_t Cols, typename D>
typename MatrixOperation<T, Rows, Cols, D>::const_reference
    MatrixOperation<T, Rows, Cols, D>::operator[](size_t i) const {
    JET_ASSERT(i < rows() * cols());
    return (*this)()[i];
}

template <typename T, size_t Rows, size_t Cols, typename D>
typename MatrixOperation<T, Rows, Cols, D>::reference
MatrixOperation<T, Rows, Cols, D>::operator()(size_t i, size_t j) {
    JET_ASSERT(i < rows() && j < cols());
    return (*this)()[j + i * cols()];
}

template <typename T, size_t Rows, size_t Cols, typename D>
typename MatrixOperation<T, Rows, Cols, D>::const_reference
MatrixOperation<T, Rows, Cols, D>::operator()(size_t i, size_t j) const {
    JET_ASSERT(i < rows() && j < cols());
    return (*this)()[j + i * cols()];
}

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename E>
MatrixOperation<T, Rows, Cols, D>& MatrixOperation<T, Rows, Cols, D>::operator=(
    const MatrixExpression<T, E>& expression) {
    copyFrom(expression);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Builders

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename U>
std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), MatrixConstant<U>>
MatrixOperation<T, Rows, Cols, D>::makeZero() {
    return MatrixConstant<T>{Rows, Cols, 0};
}

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename U>
std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), MatrixConstant<U>>
MatrixOperation<T, Rows, Cols, D>::makeZero(size_t rows, size_t cols) {
    return MatrixConstant<T>{rows, cols, 0};
}

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename U>
std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(),
                 MatrixDiagonal<U, MatrixConstant<U>>>
MatrixOperation<T, Rows, Cols, D>::makeIdentity() {
    using ConstType = MatrixConstant<U>;
    return MatrixDiagonal<U, ConstType>{ConstType{Rows, Cols, 1}};
}

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename U>
std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(),
                 MatrixDiagonal<U, MatrixConstant<U>>>
MatrixOperation<T, Rows, Cols, D>::makeIdentity(size_t rows) {
    using ConstType = MatrixConstant<U>;
    return MatrixDiagonal<U, ConstType>{ConstType{rows, rows, 1}};
}

template <typename T, size_t Rows, size_t Cols, typename Derived>
template <typename... Args, typename D>
std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
MatrixOperation<T, Rows, Cols, Derived>::makeScaleMatrix(value_type first,
                                                         Args... rest) {
    static_assert(sizeof...(rest) == Rows - 1,
                  "Number of parameters should match the size of diagonal.");
    D m{};
    std::array<T, Rows> diag{first, rest...};
    for (size_t i = 0; i < Rows; ++i) {
        m(i, i) = diag[i];
    }
    return m;
}

template <typename T, size_t Rows, size_t Cols, typename Derived>
template <typename E, typename D>
std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
MatrixOperation<T, Rows, Cols, Derived>::makeScaleMatrix(
    const MatrixExpression<T, E>& expression) {
    JET_ASSERT(expression.cols() == 1);
    D m{};
    for (size_t i = 0; i < Rows; ++i) {
        m(i, i) = expression(i, 0);
    }
    return m;
}

template <typename T, size_t Rows, size_t Cols, typename Derived>
template <typename D>
std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 2), D>
MatrixOperation<T, Rows, Cols, Derived>::makeRotationMatrix(T rad) {
    return D{std::cos(rad), -std::sin(rad), std::sin(rad), std::cos(rad)};
}

template <typename T, size_t Rows, size_t Cols, typename Derived>
template <typename E, typename D>
std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 3 || Rows == 4),
                 D>
MatrixOperation<T, Rows, Cols, Derived>::makeRotationMatrix(
    const MatrixExpression<T, E>& axis, T rad) {
    JET_ASSERT(expression.cols() == 3 || expression.cols() == 4);

    D result;

    result(0, 0) = 1 + (1 - std::cos(rad)) * (axis(0, 0) * axis(0, 0) - 1);
    result(0, 1) = -axis(2, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis(0, 0) * axis(1, 0);
    result(0, 2) = axis(1, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis(0, 0) * axis(2, 0);

    result(1, 0) = axis(2, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis(0, 0) * axis(1, 0);
    result(1, 1) = 1 + (1 - std::cos(rad)) * (axis(1, 0) * axis(1, 0) - 1);
    result(1, 2) = -axis(0, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis(1, 0) * axis(2, 0);

    result(2, 0) = -axis(1, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis(0, 0) * axis(2, 0);
    result(2, 1) = axis(0, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis(1, 0) * axis(2, 0);
    result(2, 2) = 1 + (1 - std::cos(rad)) * (axis(2, 0) * axis(2, 0) - 1);

    return result;
}

template <typename T, size_t Rows, size_t Cols, typename Derived>
template <typename E, typename D>
std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 4), D>
MatrixOperation<T, Rows, Cols, Derived>::makeTranslationMatrix(
    const MatrixExpression<T, E>& t) {
    JET_ASSERT(expression.cols() == 3);

    D result = makeIdentity();
    result(0, 3) = t(0, 0);
    result(1, 3) = t(1, 0);
    result(2, 3) = t(2, 0);

    return result;
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Private Helpers

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr size_t MatrixOperation<T, Rows, Cols, D>::rows() const {
    return static_cast<const D&>(*this).rows();
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr size_t MatrixOperation<T, Rows, Cols, D>::cols() const {
    return static_cast<const D&>(*this).cols();
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr auto MatrixOperation<T, Rows, Cols, D>::begin() {
    return (*this)().begin();
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr auto MatrixOperation<T, Rows, Cols, D>::begin() const {
    return (*this)().begin();
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr auto MatrixOperation<T, Rows, Cols, D>::end() {
    return (*this)().end();
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr auto MatrixOperation<T, Rows, Cols, D>::end() const {
    return (*this)().end();
}

template <typename T, size_t Rows, size_t Cols, typename D>
D& MatrixOperation<T, Rows, Cols, D>::operator()() {
    return static_cast<D&>(*this);
}

template <typename T, size_t Rows, size_t Cols, typename D>
const D& MatrixOperation<T, Rows, Cols, D>::operator()() const {
    return static_cast<const D&>(*this);
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr T MatrixOperation<T, Rows, Cols, D>::determinant(
    const MatrixOperation<T, 1, 1, D>& m) {
    return m(0, 0);
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr T MatrixOperation<T, Rows, Cols, D>::determinant(
    const MatrixOperation<T, 2, 2, D>& m) {
    return m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1);
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr T MatrixOperation<T, Rows, Cols, D>::determinant(
    const MatrixOperation<T, 3, 3, D>& m) {
    return m(0, 0) * m(1, 1) * m(2, 2) - m(0, 0) * m(1, 2) * m(2, 1) +
           m(0, 1) * m(1, 2) * m(2, 0) - m(0, 1) * m(1, 0) * m(2, 2) +
           m(0, 2) * m(1, 0) * m(2, 1) - m(0, 2) * m(1, 1) * m(2, 0);
}

template <typename T, size_t Rows, size_t Cols, typename D>
constexpr T MatrixOperation<T, Rows, Cols, D>::determinant(
    const MatrixOperation<T, 4, 4, D>& m) {
    return m(0, 0) * m(1, 1) * m(2, 2) * m(3, 3) +
           m(0, 0) * m(1, 2) * m(2, 3) * m(3, 1) +
           m(0, 0) * m(1, 3) * m(2, 1) * m(3, 2) +
           m(0, 1) * m(1, 0) * m(2, 3) * m(3, 2) +
           m(0, 1) * m(1, 2) * m(2, 0) * m(3, 3) +
           m(0, 1) * m(1, 3) * m(2, 2) * m(3, 0) +
           m(0, 2) * m(1, 0) * m(2, 1) * m(3, 3) +
           m(0, 2) * m(1, 1) * m(2, 3) * m(3, 0) +
           m(0, 2) * m(1, 3) * m(2, 0) * m(3, 1) +
           m(0, 3) * m(1, 0) * m(2, 2) * m(3, 1) +
           m(0, 3) * m(1, 1) * m(2, 0) * m(3, 2) +
           m(0, 3) * m(1, 2) * m(2, 1) * m(3, 0) -
           m(0, 0) * m(1, 1) * m(2, 3) * m(3, 2) -
           m(0, 0) * m(1, 2) * m(2, 1) * m(3, 3) -
           m(0, 0) * m(1, 3) * m(2, 2) * m(3, 1) -
           m(0, 1) * m(1, 0) * m(2, 2) * m(3, 3) -
           m(0, 1) * m(1, 2) * m(2, 3) * m(3, 0) -
           m(0, 1) * m(1, 3) * m(2, 0) * m(3, 2) -
           m(0, 2) * m(1, 0) * m(2, 3) * m(3, 1) -
           m(0, 2) * m(1, 1) * m(2, 0) * m(3, 3) -
           m(0, 2) * m(1, 3) * m(2, 1) * m(3, 0) -
           m(0, 3) * m(1, 0) * m(2, 1) * m(3, 2) -
           m(0, 3) * m(1, 1) * m(2, 2) * m(3, 0) -
           m(0, 3) * m(1, 2) * m(2, 0) * m(3, 1);
}

template <typename T, size_t Rows, size_t Cols, typename D>
template <typename U>
std::enable_if_t<(Rows > 4 && Cols > 4) || isMatrixSizeDynamic<Rows, Cols>(), U>
MatrixOperation<T, Rows, Cols, D>::determinant(const MatrixOperation& m) {
    // Computes inverse matrix using Gaussian elimination method.
    // https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    D a{m()};

    T result = 1;
    for (size_t i = 0; i < m.rows(); ++i) {
        // Search for maximum in this column
        T maxEl = std::fabs(a(i, i));
        size_t maxRow = i;
        for (size_t k = i + 1; k < m.rows(); ++k) {
            if (std::fabs(a(k, i)) > maxEl) {
                maxEl = std::fabs(a(k, i));
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        if (maxRow != i) {
            for (size_t k = i; k < m.rows(); ++k) {
                std::swap(a(maxRow, k), a(i, k));
            }
            result *= -1;
        }

        // Make all rows below this one 0 in current column
        for (size_t k = i + 1; k < m.rows(); ++k) {
            T c = -a(k, i) / a(i, i);
            for (size_t j = i; j < m.rows(); ++j) {
                if (i == j) {
                    a(k, j) = 0;
                } else {
                    a(k, j) += c * a(i, j);
                }
            }
        }
    }

    for (size_t i = 0; i < m.rows(); ++i) {
        result *= a(i, i);
    }
    return result;
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::inverse(
    const MatrixOperation<T, 1, 1, D>& m, D& result) {
    result[0] = 1 / m[0];
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::inverse(
    const MatrixOperation<T, 2, 2, D>& m, D& result) {
    T d = determinant(m);
    result(0, 0) = m(1, 1) / d;
    result(0, 1) = -m(0, 1) / d;
    result(1, 0) = -m(1, 0) / d;
    result(1, 1) = m(0, 0) / d;
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::inverse(
    const MatrixOperation<T, 3, 3, D>& m, D& result) {
    T d = determinant(m);

    result(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / d;
    result(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) / d;
    result(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / d;
    result(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) / d;
    result(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / d;
    result(1, 2) = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) / d;
    result(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) / d;
    result(2, 1) = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) / d;
    result(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) / d;
}

template <typename T, size_t Rows, size_t Cols, typename D>
void MatrixOperation<T, Rows, Cols, D>::inverse(
    const MatrixOperation<T, 4, 4, D>& m, D& result) {
    T d = determinant(m);
    result(0, 0) = (m(1, 1) * m(2, 2) * m(3, 3) + m(1, 2) * m(2, 3) * m(3, 1) +
                    m(1, 3) * m(2, 1) * m(3, 2) - m(1, 1) * m(2, 3) * m(3, 2) -
                    m(1, 2) * m(2, 1) * m(3, 3) - m(1, 3) * m(2, 2) * m(3, 1)) /
                   d;
    result(0, 1) = (m(0, 1) * m(2, 3) * m(3, 2) + m(0, 2) * m(2, 1) * m(3, 3) +
                    m(0, 3) * m(2, 2) * m(3, 1) - m(0, 1) * m(2, 2) * m(3, 3) -
                    m(0, 2) * m(2, 3) * m(3, 1) - m(0, 3) * m(2, 1) * m(3, 2)) /
                   d;
    result(0, 2) = (m(0, 1) * m(1, 2) * m(3, 3) + m(0, 2) * m(1, 3) * m(3, 1) +
                    m(0, 3) * m(1, 1) * m(3, 2) - m(0, 1) * m(1, 3) * m(3, 2) -
                    m(0, 2) * m(1, 1) * m(3, 3) - m(0, 3) * m(1, 2) * m(3, 1)) /
                   d;
    result(0, 3) = (m(0, 1) * m(1, 3) * m(2, 2) + m(0, 2) * m(1, 1) * m(2, 3) +
                    m(0, 3) * m(1, 2) * m(2, 1) - m(0, 1) * m(1, 2) * m(2, 3) -
                    m(0, 2) * m(1, 3) * m(2, 1) - m(0, 3) * m(1, 1) * m(2, 2)) /
                   d;
    result(1, 0) = (m(1, 0) * m(2, 3) * m(3, 2) + m(1, 2) * m(2, 0) * m(3, 3) +
                    m(1, 3) * m(2, 2) * m(3, 0) - m(1, 0) * m(2, 2) * m(3, 3) -
                    m(1, 2) * m(2, 3) * m(3, 0) - m(1, 3) * m(2, 0) * m(3, 2)) /
                   d;
    result(1, 1) = (m(0, 0) * m(2, 2) * m(3, 3) + m(0, 2) * m(2, 3) * m(3, 0) +
                    m(0, 3) * m(2, 0) * m(3, 2) - m(0, 0) * m(2, 3) * m(3, 2) -
                    m(0, 2) * m(2, 0) * m(3, 3) - m(0, 3) * m(2, 2) * m(3, 0)) /
                   d;
    result(1, 2) = (m(0, 0) * m(1, 3) * m(3, 2) + m(0, 2) * m(1, 0) * m(3, 3) +
                    m(0, 3) * m(1, 2) * m(3, 0) - m(0, 0) * m(1, 2) * m(3, 3) -
                    m(0, 2) * m(1, 3) * m(3, 0) - m(0, 3) * m(1, 0) * m(3, 2)) /
                   d;
    result(1, 3) = (m(0, 0) * m(1, 2) * m(2, 3) + m(0, 2) * m(1, 3) * m(2, 0) +
                    m(0, 3) * m(1, 0) * m(2, 2) - m(0, 0) * m(1, 3) * m(2, 2) -
                    m(0, 2) * m(1, 0) * m(2, 3) - m(0, 3) * m(1, 2) * m(2, 0)) /
                   d;
    result(2, 0) = (m(1, 0) * m(2, 1) * m(3, 3) + m(1, 1) * m(2, 3) * m(3, 0) +
                    m(1, 3) * m(2, 0) * m(3, 1) - m(1, 0) * m(2, 3) * m(3, 1) -
                    m(1, 1) * m(2, 0) * m(3, 3) - m(1, 3) * m(2, 1) * m(3, 0)) /
                   d;
    result(2, 1) = (m(0, 0) * m(2, 3) * m(3, 1) + m(0, 1) * m(2, 0) * m(3, 3) +
                    m(0, 3) * m(2, 1) * m(3, 0) - m(0, 0) * m(2, 1) * m(3, 3) -
                    m(0, 1) * m(2, 3) * m(3, 0) - m(0, 3) * m(2, 0) * m(3, 1)) /
                   d;
    result(2, 2) = (m(0, 0) * m(1, 1) * m(3, 3) + m(0, 1) * m(1, 3) * m(3, 0) +
                    m(0, 3) * m(1, 0) * m(3, 1) - m(0, 0) * m(1, 3) * m(3, 1) -
                    m(0, 1) * m(1, 0) * m(3, 3) - m(0, 3) * m(1, 1) * m(3, 0)) /
                   d;
    result(2, 3) = (m(0, 0) * m(1, 3) * m(2, 1) + m(0, 1) * m(1, 0) * m(2, 3) +
                    m(0, 3) * m(1, 1) * m(2, 0) - m(0, 0) * m(1, 1) * m(2, 3) -
                    m(0, 1) * m(1, 3) * m(2, 0) - m(0, 3) * m(1, 0) * m(2, 1)) /
                   d;
    result(3, 0) = (m(1, 0) * m(2, 2) * m(3, 1) + m(1, 1) * m(2, 0) * m(3, 2) +
                    m(1, 2) * m(2, 1) * m(3, 0) - m(1, 0) * m(2, 1) * m(3, 2) -
                    m(1, 1) * m(2, 2) * m(3, 0) - m(1, 2) * m(2, 0) * m(3, 1)) /
                   d;
    result(3, 1) = (m(0, 0) * m(2, 1) * m(3, 2) + m(0, 1) * m(2, 2) * m(3, 0) +
                    m(0, 2) * m(2, 0) * m(3, 1) - m(0, 0) * m(2, 2) * m(3, 1) -
                    m(0, 1) * m(2, 0) * m(3, 2) - m(0, 2) * m(2, 1) * m(3, 0)) /
                   d;
    result(3, 2) = (m(0, 0) * m(1, 2) * m(3, 1) + m(0, 1) * m(1, 0) * m(3, 2) +
                    m(0, 2) * m(1, 1) * m(3, 0) - m(0, 0) * m(1, 1) * m(3, 2) -
                    m(0, 1) * m(1, 2) * m(3, 0) - m(0, 2) * m(1, 0) * m(3, 1)) /
                   d;
    result(3, 3) = (m(0, 0) * m(1, 1) * m(2, 2) + m(0, 1) * m(1, 2) * m(2, 0) +
                    m(0, 2) * m(1, 0) * m(2, 1) - m(0, 0) * m(1, 2) * m(2, 1) -
                    m(0, 1) * m(1, 0) * m(2, 2) - m(0, 2) * m(1, 1) * m(2, 0)) /
                   d;
}

template <typename T, size_t Rows, size_t Cols, typename Derived>
template <typename D>
void MatrixOperation<T, Rows, Cols, Derived>::inverse(
    const MatrixOperation<T, Rows, Cols, Derived>& m,
    std::enable_if_t<(Rows > 4 && Cols > 4) ||
                         isMatrixSizeDynamic<Rows, Cols>(),
                     D>& result) {
    // Computes inverse matrix using Gaussian elimination method.
    // https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    D a{m()};

    using ConstType = MatrixConstant<T>;
    result = MatrixDiagonal<T, ConstType>{ConstType{a.rows(), a.cols(), 1}};
    size_t n = m.rows();

    for (size_t i = 0; i < n; ++i) {
        // Search for maximum in this column
        T maxEl = std::fabs(a(i, i));
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; ++k) {
            if (std::fabs(a(k, i)) > maxEl) {
                maxEl = std::fabs(a(k, i));
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        if (maxRow != i) {
            for (size_t k = i; k < n; ++k) {
                std::swap(a(maxRow, k), a(i, k));
            }
            for (size_t k = 0; k < n; ++k) {
                std::swap(result(maxRow, k), result(i, k));
            }
        }

        // Make all rows except this one 0 in current column
        for (size_t k = 0; k < n; ++k) {
            if (k == i) {
                continue;
            }
            T c = -a(k, i) / a(i, i);
            for (size_t j = 0; j < n; ++j) {
                result(k, j) += c * result(i, j);
                if (i == j) {
                    a(k, j) = 0;
                } else if (i < j) {
                    a(k, j) += c * a(i, j);
                }
            }
        }

        // Scale
        for (size_t k = 0; k < n; ++k) {
            T c = 1 / a(k, k);
            for (size_t j = 0; j < n; ++j) {
                a(k, j) *= c;
                result(k, j) *= c;
            }
        }
    }
}

}  // namespace jet
#endif  // INCLUDE_JET_DETAIL_MATRIX_OPERATION_INL_H_
