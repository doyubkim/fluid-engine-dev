// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_MATRIX_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX_INL_H_

#include <jet/macros.h>
#include <jet/math_utils.h>

namespace jet {

template <typename T, size_t M, size_t N>
Matrix<T, M, N>::Matrix() {
    for (auto& elem : _elements) {
        elem = 0;
    }
}

template <typename T, size_t M, size_t N>
template <typename... Params>
Matrix<T, M, N>::Matrix(Params... params) {
    static_assert(sizeof...(params) == M * N, "Invalid number of elements.");

    setRowAt(0, params...);
}

template <typename T, size_t M, size_t N>
Matrix<T, M, N>::Matrix(
    const std::initializer_list<std::initializer_list<T>>& lst) {
    set(lst);
}

template <typename T, size_t M, size_t N>
template <typename E>
Matrix<T, M, N>::Matrix(const MatrixExpression<T, E>& other) {
    set(other);
}

template <typename T, size_t M, size_t N>
Matrix<T, M, N>::Matrix(const Matrix& other) {
    set(other);
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::set(const T& s) {
    _elements.fill(s);
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::set(
    const std::initializer_list<std::initializer_list<T>>& lst) {
    size_t rows = lst.size();
    size_t cols = (rows > 0) ? lst.begin()->size() : 0;

    JET_ASSERT(rows == M);
    JET_ASSERT(cols == N);

    auto rowIter = lst.begin();
    for (size_t i = 0; i < rows; ++i) {
        JET_ASSERT(cols == rowIter->size());
        auto colIter = rowIter->begin();
        for (size_t j = 0; j < cols; ++j) {
            (*this)(i, j) = *colIter;
            ++colIter;
        }
        ++rowIter;
    }
}

template <typename T, size_t M, size_t N>
template <typename E>
void Matrix<T, M, N>::set(const MatrixExpression<T, E>& other) {
    const E& expression = other();
    forEachIndex([&](size_t i, size_t j) { (*this)(i, j) = expression(i, j); });
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::setDiagonal(const T& s) {
    const size_t l = std::min(rows(), cols());
    for (size_t i = 0; i < l; ++i) {
        (*this)(i, i) = s;
    }
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::setOffDiagonal(const T& s) {
    forEachIndex([&](size_t i, size_t j) {
        if (i != j) {
            (*this)(i, j) = s;
        }
    });
}

template <typename T, size_t M, size_t N>
template <typename E>
void Matrix<T, M, N>::setRow(size_t i, const VectorExpression<T, E>& row) {
    JET_ASSERT(cols() == row.size());

    const E& e = row();
    for (size_t j = 0; j < N; ++j) {
        (*this)(i, j) = e[j];
    }
}

template <typename T, size_t M, size_t N>
template <typename E>
void Matrix<T, M, N>::setColumn(size_t j, const VectorExpression<T, E>& col) {
    JET_ASSERT(rows() == col.size());

    const E& e = col();
    for (size_t i = 0; i < M; ++i) {
        (*this)(i, j) = e[j];
    }
}

template <typename T, size_t M, size_t N>
template <typename E>
bool Matrix<T, M, N>::isEqual(const MatrixExpression<T, E>& other) const {
    if (size() != other.size()) {
        return false;
    }

    const E& e = other();
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            if ((*this)(i, j) != e(i, j)) {
                return false;
            }
        }
    }

    return true;
}

template <typename T, size_t M, size_t N>
template <typename E>
bool Matrix<T, M, N>::isSimilar(const MatrixExpression<T, E>& other,
                                double tol) const {
    if (size() != other.size()) {
        return false;
    }

    const E& e = other();
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            if (std::fabs((*this)(i, j) - e(i, j)) > tol) {
                return false;
            }
        }
    }

    return true;
}

template <typename T, size_t M, size_t N>
constexpr bool Matrix<T, M, N>::isSquare() const {
    return M == N;
}

template <typename T, size_t M, size_t N>
constexpr Size2 Matrix<T, M, N>::size() const {
    return Size2(M, N);
}

template <typename T, size_t M, size_t N>
constexpr size_t Matrix<T, M, N>::rows() const {
    return M;
}

template <typename T, size_t M, size_t N>
constexpr size_t Matrix<T, M, N>::cols() const {
    return N;
}

template <typename T, size_t M, size_t N>
T* Matrix<T, M, N>::data() {
    return _elements.data();
}

template <typename T, size_t M, size_t N>
const T* const Matrix<T, M, N>::data() const {
    return _elements.data();
}

template <typename T, size_t M, size_t N>
typename Matrix<T, M, N>::Iterator Matrix<T, M, N>::begin() {
    return _elements.begin();
}

template <typename T, size_t M, size_t N>
typename Matrix<T, M, N>::ConstIterator Matrix<T, M, N>::begin() const {
    return _elements.begin();
}

template <typename T, size_t M, size_t N>
typename Matrix<T, M, N>::Iterator Matrix<T, M, N>::end() {
    return _elements.end();
}

template <typename T, size_t M, size_t N>
typename Matrix<T, M, N>::ConstIterator Matrix<T, M, N>::end() const {
    return _elements.end();
}

template <typename T, size_t M, size_t N>
MatrixScalarAdd<T, Matrix<T, M, N>> Matrix<T, M, N>::add(const T& s) const {
    return MatrixScalarAdd<T, Matrix<T, M, N>>(*this, s);
}

template <typename T, size_t M, size_t N>
template <typename E>
MatrixAdd<T, Matrix<T, M, N>, E> Matrix<T, M, N>::add(const E& m) const {
    return MatrixAdd<T, Matrix, E>(*this, m);
}

template <typename T, size_t M, size_t N>
MatrixScalarSub<T, Matrix<T, M, N>> Matrix<T, M, N>::sub(const T& s) const {
    return MatrixScalarSub<T, Matrix<T, M, N>>(*this, s);
}

template <typename T, size_t M, size_t N>
template <typename E>
MatrixSub<T, Matrix<T, M, N>, E> Matrix<T, M, N>::sub(const E& m) const {
    return MatrixSub<T, Matrix, E>(*this, m);
}

template <typename T, size_t M, size_t N>
MatrixScalarMul<T, Matrix<T, M, N>> Matrix<T, M, N>::mul(const T& s) const {
    return MatrixScalarMul<T, Matrix>(*this, s);
}

template <typename T, size_t M, size_t N>
template <typename VE>
MatrixVectorMul<T, Matrix<T, M, N>, VE> Matrix<T, M, N>::mul(
    const VectorExpression<T, VE>& v) const {
    return MatrixVectorMul<T, Matrix<T, M, N>, VE>(*this, v());
}

template <typename T, size_t M, size_t N>
template <size_t L>
MatrixMul<T, Matrix<T, M, N>, Matrix<T, N, L>> Matrix<T, M, N>::mul(
    const Matrix<T, N, L>& m) const {
    return MatrixMul<T, Matrix, Matrix<T, N, L>>(*this, m);
}

template <typename T, size_t M, size_t N>
MatrixScalarDiv<T, Matrix<T, M, N>> Matrix<T, M, N>::div(const T& s) const {
    return MatrixScalarDiv<T, Matrix>(*this, s);
}

template <typename T, size_t M, size_t N>
MatrixScalarAdd<T, Matrix<T, M, N>> Matrix<T, M, N>::radd(const T& s) const {
    return MatrixScalarAdd<T, Matrix<T, M, N>>(*this, s);
}

template <typename T, size_t M, size_t N>
template <typename E>
MatrixAdd<T, Matrix<T, M, N>, E> Matrix<T, M, N>::radd(const E& m) const {
    return MatrixAdd<T, Matrix<T, M, N>, E>(m, *this);
}

template <typename T, size_t M, size_t N>
MatrixScalarRSub<T, Matrix<T, M, N>> Matrix<T, M, N>::rsub(const T& s) const {
    return MatrixScalarRSub<T, Matrix<T, M, N>>(*this, s);
}

template <typename T, size_t M, size_t N>
template <typename E>
MatrixSub<T, Matrix<T, M, N>, E> Matrix<T, M, N>::rsub(const E& m) const {
    return MatrixSub<T, Matrix<T, M, N>, E>(m, *this);
}

template <typename T, size_t M, size_t N>
MatrixScalarMul<T, Matrix<T, M, N>> Matrix<T, M, N>::rmul(const T& s) const {
    return MatrixScalarMul<T, Matrix<T, M, N>>(*this, s);
}

template <typename T, size_t M, size_t N>
template <size_t L>
MatrixMul<T, Matrix<T, N, L>, Matrix<T, M, N>> Matrix<T, M, N>::rmul(
    const Matrix<T, N, L>& m) const {
    return MatrixMul<T, Matrix<T, N, L>, Matrix>(m, *this);
}

template <typename T, size_t M, size_t N>
MatrixScalarRDiv<T, Matrix<T, M, N>> Matrix<T, M, N>::rdiv(const T& s) const {
    return MatrixScalarRDiv<T, Matrix<T, M, N>>(*this, s);
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::iadd(const T& s) {
    set(add(s));
}

template <typename T, size_t M, size_t N>
template <typename E>
void Matrix<T, M, N>::iadd(const E& m) {
    set(add(m));
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::isub(const T& s) {
    set(sub(s));
}

template <typename T, size_t M, size_t N>
template <typename E>
void Matrix<T, M, N>::isub(const E& m) {
    set(sub(m));
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::imul(const T& s) {
    set(mul(s));
}

template <typename T, size_t M, size_t N>
template <typename E>
void Matrix<T, M, N>::imul(const E& m) {
    Matrix tmp = mul(m);
    set(tmp);
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::idiv(const T& s) {
    set(div(s));
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::transpose() {
    set(transposed());
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::invert() {
    JET_ASSERT(isSquare());

    // Computes inverse matrix using Gaussian elimination method.
    // https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    size_t n = rows();
    Matrix& a = *this;
    Matrix rhs = makeIdentity();

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
                std::swap(rhs(maxRow, k), rhs(i, k));
            }
        }

        // Make all rows except this one 0 in current column
        for (size_t k = 0; k < n; ++k) {
            if (k == i) {
                continue;
            }
            T c = -a(k, i) / a(i, i);
            for (size_t j = 0; j < n; ++j) {
                rhs(k, j) += c * rhs(i, j);
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
                rhs(k, j) *= c;
            }
        }
    }

    set(rhs);
}

template <typename T, size_t M, size_t N>
T Matrix<T, M, N>::sum() const {
    T ret = 0;
    for (auto v : _elements) {
        ret += v;
    }
    return ret;
}

template <typename T, size_t M, size_t N>
T Matrix<T, M, N>::avg() const {
    return sum() / (rows() * cols());
}

template <typename T, size_t M, size_t N>
T Matrix<T, M, N>::min() const {
    T ret = _elements.front();
    for (auto v : _elements) {
        ret = std::min(ret, v);
    }
    return ret;
}

template <typename T, size_t M, size_t N>
T Matrix<T, M, N>::max() const {
    T ret = _elements.front();
    for (auto v : _elements) {
        ret = std::max(ret, v);
    }
    return ret;
}

template <typename T, size_t M, size_t N>
T Matrix<T, M, N>::absmin() const {
    T ret = _elements.front();
    for (auto v : _elements) {
        ret = jet::absmin(ret, v);
    }
    return ret;
}

template <typename T, size_t M, size_t N>
T Matrix<T, M, N>::absmax() const {
    T ret = _elements.front();
    for (auto v : _elements) {
        ret = jet::absmax(ret, v);
    }
    return ret;
}

template <typename T, size_t M, size_t N>
T Matrix<T, M, N>::trace() const {
    JET_ASSERT(isSquare());
    T ret = 0;
    for (size_t i = 0; i < M; ++i) {
        ret += (*this)(i, i);
    }
    return ret;
}

template <typename T, size_t M, size_t N>
T Matrix<T, M, N>::determinant() const {
    JET_ASSERT(isSquare());

    // Computes inverse matrix using Gaussian elimination method.
    // https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    size_t n = rows();
    Matrix a(*this);

    T result = 1;
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
            result *= -1;
        }

        // Make all rows below this one 0 in current column
        for (size_t k = i + 1; k < n; ++k) {
            T c = -a(k, i) / a(i, i);
            for (size_t j = i; j < n; ++j) {
                if (i == j) {
                    a(k, j) = 0;
                } else {
                    a(k, j) += c * a(i, j);
                }
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        result *= a(i, i);
    }
    return result;
}

template <typename T, size_t M, size_t N>
MatrixDiagonal<T, Matrix<T, M, N>> Matrix<T, M, N>::diagonal() const {
    return MatrixDiagonal<T, Matrix>(*this, true);
}

template <typename T, size_t M, size_t N>
MatrixDiagonal<T, Matrix<T, M, N>> Matrix<T, M, N>::offDiagonal() const {
    return MatrixDiagonal<T, Matrix>(*this, false);
}

template <typename T, size_t M, size_t N>
MatrixTriangular<T, Matrix<T, M, N>> Matrix<T, M, N>::strictLowerTri() const {
    return MatrixTriangular<T, Matrix<T, M, N>>(*this, false, true);
}

template <typename T, size_t M, size_t N>
MatrixTriangular<T, Matrix<T, M, N>> Matrix<T, M, N>::strictUpperTri() const {
    return MatrixTriangular<T, Matrix<T, M, N>>(*this, true, true);
}

template <typename T, size_t M, size_t N>
MatrixTriangular<T, Matrix<T, M, N>> Matrix<T, M, N>::lowerTri() const {
    return MatrixTriangular<T, Matrix<T, M, N>>(*this, false, false);
}

template <typename T, size_t M, size_t N>
MatrixTriangular<T, Matrix<T, M, N>> Matrix<T, M, N>::upperTri() const {
    return MatrixTriangular<T, Matrix<T, M, N>>(*this, true, false);
}

template <typename T, size_t M, size_t N>
Matrix<T, N, M> Matrix<T, M, N>::transposed() const {
    Matrix<T, N, M> mt;
    forEachIndex([&](size_t i, size_t j) { mt(j, i) = (*this)(i, j); });
    return mt;
}

template <typename T, size_t M, size_t N>
Matrix<T, M, N> Matrix<T, M, N>::inverse() const {
    Matrix mInv(*this);
    mInv.invert();
    return mInv;
}

template <typename T, size_t M, size_t N>
template <typename U>
MatrixTypeCast<U, Matrix<T, M, N>, T> Matrix<T, M, N>::castTo() const {
    return MatrixTypeCast<U, Matrix, T>(*this);
}

template <typename T, size_t M, size_t N>
template <typename E>
Matrix<T, M, N>& Matrix<T, M, N>::operator=(const E& m) {
    set(m);
    return *this;
}

template <typename T, size_t M, size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator=(const Matrix& other) {
    set(other);
    return *this;
}

template <typename T, size_t M, size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator+=(const T& s) {
    iadd(s);
    return *this;
}

template <typename T, size_t M, size_t N>
template <typename E>
Matrix<T, M, N>& Matrix<T, M, N>::operator+=(const E& m) {
    iadd(m);
    return *this;
}

template <typename T, size_t M, size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator-=(const T& s) {
    isub(s);
    return *this;
}

template <typename T, size_t M, size_t N>
template <typename E>
Matrix<T, M, N>& Matrix<T, M, N>::operator-=(const E& m) {
    isub(m);
    return *this;
}

template <typename T, size_t M, size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator*=(const T& s) {
    imul(s);
    return *this;
}

template <typename T, size_t M, size_t N>
template <typename E>
Matrix<T, M, N>& Matrix<T, M, N>::operator*=(const E& m) {
    imul(m);
    return *this;
}

template <typename T, size_t M, size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator/=(const T& s) {
    idiv(s);
    return *this;
}

template <typename T, size_t M, size_t N>
T& Matrix<T, M, N>::operator[](size_t i) {
    return _elements[i];
}

template <typename T, size_t M, size_t N>
const T& Matrix<T, M, N>::operator[](size_t i) const {
    return _elements[i];
}

template <typename T, size_t M, size_t N>
T& Matrix<T, M, N>::operator()(size_t i, size_t j) {
    return _elements[i * N + j];
}

template <typename T, size_t M, size_t N>
const T& Matrix<T, M, N>::operator()(size_t i, size_t j) const {
    return _elements[i * N + j];
}

template <typename T, size_t M, size_t N>
template <typename E>
bool Matrix<T, M, N>::operator==(const MatrixExpression<T, E>& m) const {
    return isEqual(m);
}

template <typename T, size_t M, size_t N>
template <typename E>
bool Matrix<T, M, N>::operator!=(const MatrixExpression<T, E>& m) const {
    return !isEqual(m);
}

template <typename T, size_t M, size_t N>
template <typename Callback>
void Matrix<T, M, N>::forEach(Callback func) const {
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            func((*this)(i, j));
        }
    }
}

template <typename T, size_t M, size_t N>
template <typename Callback>
void Matrix<T, M, N>::forEachIndex(Callback func) const {
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            func(i, j);
        }
    }
}

template <typename T, size_t M, size_t N>
MatrixConstant<T> Matrix<T, M, N>::makeZero() {
    return MatrixConstant<T>(M, N, 0);
}

template <typename T, size_t M, size_t N>
MatrixIdentity<T> Matrix<T, M, N>::makeIdentity() {
    static_assert(M == N, "Should be a square matrix.");
    return MatrixIdentity<T>(M);
}

template <typename T, size_t M, size_t N>
template <typename... Params>
void Matrix<T, M, N>::setRowAt(size_t i, T v, Params... params) {
    _elements[i] = v;
    setRowAt(i + 1, params...);
}

template <typename T, size_t M, size_t N>
void Matrix<T, M, N>::setRowAt(size_t i, T v) {
    _elements[i] = v;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_MATRIX_INL_H_
