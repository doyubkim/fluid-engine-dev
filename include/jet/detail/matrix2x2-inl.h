// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_MATRIX2X2_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX2X2_INL_H_

#include <jet/math_utils.h>
#include <algorithm>
#include <cstring>
#include <utility>

namespace jet {

// MARK: CTOR/DTOR
template <typename T>
Matrix<T, 2, 2>::Matrix() {
    set(1, 0, 0, 1);
}

template <typename T>
Matrix<T, 2, 2>::Matrix(T s) {
    set(s);
}

template <typename T>
Matrix<T, 2, 2>::Matrix(T m00, T m01, T m10, T m11) {
    set(m00, m01, m10, m11);
}

template <typename T>
template <typename U>
Matrix<T, 2, 2>::Matrix(
    const std::initializer_list<std::initializer_list<U>>& lst) {
    set(lst);
}

template <typename T>
Matrix<T, 2, 2>::Matrix(const Matrix& m) {
    set(m);
}

template <typename T>
Matrix<T, 2, 2>::Matrix(const T* arr) {
    set(arr);
}

// MARK: Basic setters
template <typename T>
void Matrix<T, 2, 2>::set(T s) {
    _elements[0] = _elements[1] = _elements[2] = _elements[3] = s;
}

template <typename T>
void Matrix<T, 2, 2>::set(T m00, T m01, T m10, T m11) {
    _elements[0] = m00;
    _elements[1] = m01;
    _elements[2] = m10;
    _elements[3] = m11;
}

template <typename T>
template <typename U>
void Matrix<T, 2, 2>::set(
    const std::initializer_list<std::initializer_list<U>>& lst) {
    size_t height = lst.size();
    size_t width = (height > 0) ? lst.begin()->size() : 0;
    JET_ASSERT(width == 2);
    JET_ASSERT(height == 2);

    auto rowIter = lst.begin();
    for (size_t i = 0; i < height; ++i) {
        JET_ASSERT(width == rowIter->size());
        auto colIter = rowIter->begin();
        for (size_t j = 0; j < width; ++j) {
            (*this)(i, j) = static_cast<T>(*colIter);
            ++colIter;
        }
        ++rowIter;
    }
}

template <typename T>
void Matrix<T, 2, 2>::set(const Matrix& m) {
    memcpy(_elements.data(), m._elements.data(), sizeof(T) * 4);
}

template <typename T>
void Matrix<T, 2, 2>::set(const T* arr) {
    memcpy(_elements.data(), arr, sizeof(T) * 4);
}

template <typename T>
void Matrix<T, 2, 2>::setDiagonal(T s) {
    _elements[0] = _elements[3] = s;
}

template <typename T>
void Matrix<T, 2, 2>::setOffDiagonal(T s) {
    _elements[1] = _elements[2] = s;
}

template <typename T>
void Matrix<T, 2, 2>::setRow(size_t i, const Vector<T, 2>& row) {
    _elements[2 * i] = row.x;
    _elements[2 * i + 1] = row.y;
}

template <typename T>
void Matrix<T, 2, 2>::setColumn(size_t j, const Vector<T, 2>& col) {
    _elements[j] = col.x;
    _elements[j + 2] = col.y;
}

// MARK: Basic getters
template <typename T>
bool Matrix<T, 2, 2>::isSimilar(const Matrix& m, double tol) const {
    return (std::fabs(_elements[0] - m._elements[0]) < tol) &&
           (std::fabs(_elements[1] - m._elements[1]) < tol) &&
           (std::fabs(_elements[2] - m._elements[2]) < tol) &&
           (std::fabs(_elements[3] - m._elements[3]) < tol);
}

template <typename T>
bool Matrix<T, 2, 2>::isSquare() const {
    return true;
}

template <typename T>
size_t Matrix<T, 2, 2>::rows() const {
    return 2;
}

template <typename T>
size_t Matrix<T, 2, 2>::cols() const {
    return 2;
}

template <typename T>
T* Matrix<T, 2, 2>::data() {
    return _elements.data();
}

template <typename T>
const T* Matrix<T, 2, 2>::data() const {
    return _elements.data();
}

// MARK: Binary operator methods - new instance = this instance (+) input
template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::add(T s) const {
    return Matrix(_elements[0] + s, _elements[1] + s, _elements[2] + s,
                  _elements[3] + s);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::add(const Matrix& m) const {
    return Matrix(_elements[0] + m._elements[0], _elements[1] + m._elements[1],
                  _elements[2] + m._elements[2], _elements[3] + m._elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::sub(T s) const {
    return Matrix(_elements[0] - s, _elements[1] - s, _elements[2] - s,
                  _elements[3] - s);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::sub(const Matrix& m) const {
    return Matrix(_elements[0] - m._elements[0], _elements[1] - m._elements[1],
                  _elements[2] - m._elements[2], _elements[3] - m._elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::mul(T s) const {
    return Matrix(_elements[0] * s, _elements[1] * s, _elements[2] * s,
                  _elements[3] * s);
}

template <typename T>
Vector<T, 2> Matrix<T, 2, 2>::mul(const Vector<T, 2>& v) const {
    return Vector<T, 2>(_elements[0] * v.x + _elements[1] * v.y,
                        _elements[2] * v.x + _elements[3] * v.y);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::mul(const Matrix& m) const {
    return Matrix(
        _elements[0] * m._elements[0] + _elements[1] * m._elements[2],
        _elements[0] * m._elements[1] + _elements[1] * m._elements[3],
        _elements[2] * m._elements[0] + _elements[3] * m._elements[2],
        _elements[2] * m._elements[1] + _elements[3] * m._elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::div(T s) const {
    return Matrix(_elements[0] / s, _elements[1] / s, _elements[2] / s,
                  _elements[3] / s);
}

// MARK: Binary operator methods - new instance = input (+) this instance
template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::radd(T s) const {
    return Matrix(s + _elements[0], s + _elements[1], s + _elements[2],
                  s + _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::radd(const Matrix& m) const {
    return Matrix(m._elements[0] + _elements[0], m._elements[1] + _elements[1],
                  m._elements[2] + _elements[2], m._elements[3] + _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rsub(T s) const {
    return Matrix(s - _elements[0], s - _elements[1], s - _elements[2],
                  s - _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rsub(const Matrix& m) const {
    return Matrix(m._elements[0] - _elements[0], m._elements[1] - _elements[1],
                  m._elements[2] - _elements[2], m._elements[3] - _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rmul(T s) const {
    return Matrix(s * _elements[0], s * _elements[1], s * _elements[2],
                  s * _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rmul(const Matrix& m) const {
    return m.mul(*this);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rdiv(T s) const {
    return Matrix(s / _elements[0], s / _elements[1], s / _elements[2],
                  s / _elements[3]);
}

// MARK: Augmented operator methods - this instance (+)= input
template <typename T>
void Matrix<T, 2, 2>::iadd(T s) {
    _elements[0] += s;
    _elements[1] += s;
    _elements[2] += s;
    _elements[3] += s;
}

template <typename T>
void Matrix<T, 2, 2>::iadd(const Matrix& m) {
    _elements[0] += m._elements[0];
    _elements[1] += m._elements[1];
    _elements[2] += m._elements[2];
    _elements[3] += m._elements[3];
}

template <typename T>
void Matrix<T, 2, 2>::isub(T s) {
    _elements[0] -= s;
    _elements[1] -= s;
    _elements[2] -= s;
    _elements[3] -= s;
}

template <typename T>
void Matrix<T, 2, 2>::isub(const Matrix& m) {
    _elements[0] -= m._elements[0];
    _elements[1] -= m._elements[1];
    _elements[2] -= m._elements[2];
    _elements[3] -= m._elements[3];
}

template <typename T>
void Matrix<T, 2, 2>::imul(T s) {
    _elements[0] *= s;
    _elements[1] *= s;
    _elements[2] *= s;
    _elements[3] *= s;
}

template <typename T>
void Matrix<T, 2, 2>::imul(const Matrix& m) {
    set(mul(m));
}

template <typename T>
void Matrix<T, 2, 2>::idiv(T s) {
    _elements[0] /= s;
    _elements[1] /= s;
    _elements[2] /= s;
    _elements[3] /= s;
}

// MARK: Modifiers
template <typename T>
void Matrix<T, 2, 2>::transpose() {
    std::swap(_elements[1], _elements[2]);
}

template <typename T>
void Matrix<T, 2, 2>::invert() {
    T d = determinant();
    Matrix m;
    m._elements[0] = _elements[3];
    m._elements[1] = -_elements[1];
    m._elements[2] = -_elements[2];
    m._elements[3] = _elements[0];
    m.idiv(d);

    set(m);
}

// MARK: Complex getters
template <typename T>
T Matrix<T, 2, 2>::sum() const {
    T s = 0;
    for (int i = 0; i < 4; ++i) {
        s += _elements[i];
    }
    return s;
}

template <typename T>
T Matrix<T, 2, 2>::avg() const {
    return sum() / 4;
}

template <typename T>
T Matrix<T, 2, 2>::min() const {
    return std::min(std::min(_elements[0], _elements[1]),
                    std::min(_elements[2], _elements[3]));
}

template <typename T>
T Matrix<T, 2, 2>::max() const {
    return std::max(std::max(_elements[0], _elements[1]),
                    std::max(_elements[2], _elements[3]));
}

template <typename T>
T Matrix<T, 2, 2>::absmin() const {
    return ::jet::absmin(::jet::absmin(_elements[0], _elements[1]),
                         ::jet::absmin(_elements[2], _elements[3]));
}

template <typename T>
T Matrix<T, 2, 2>::absmax() const {
    return ::jet::absmax(::jet::absmax(_elements[0], _elements[1]),
                         ::jet::absmax(_elements[2], _elements[3]));
}

template <typename T>
T Matrix<T, 2, 2>::trace() const {
    return _elements[0] + _elements[3];
}

template <typename T>
T Matrix<T, 2, 2>::determinant() const {
    return _elements[0] * _elements[3] - _elements[1] * _elements[2];
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::diagonal() const {
    return Matrix(_elements[0], 0, 0, _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::offDiagonal() const {
    return Matrix(0, _elements[1], _elements[2], 0);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::strictLowerTri() const {
    return Matrix(0, 0, _elements[2], 0);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::strictUpperTri() const {
    return Matrix(0, _elements[1], 0, 0);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::lowerTri() const {
    return Matrix(_elements[0], 0, _elements[2], _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::upperTri() const {
    return Matrix(_elements[0], _elements[1], 0, _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::transposed() const {
    return Matrix(_elements[0], _elements[2], _elements[1], _elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::inverse() const {
    Matrix m(*this);
    m.invert();
    return m;
}

template <typename T>
T Matrix<T, 2, 2>::frobeniusNorm() const {
    return std::sqrt(_elements[0] * _elements[0] + _elements[1] * _elements[1] +
                     _elements[2] * _elements[2] + _elements[3] * _elements[3]);
}

template <typename T>
template <typename U>
Matrix<U, 2, 2> Matrix<T, 2, 2>::castTo() const {
    return Matrix<U, 2, 2>(
        static_cast<U>(_elements[0]), static_cast<U>(_elements[1]),
        static_cast<U>(_elements[2]), static_cast<U>(_elements[3]));
}

// MARK: Setter operators
template <typename T>
Matrix<T, 2, 2>& Matrix<T, 2, 2>::operator=(const Matrix& m) {
    set(m);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>& Matrix<T, 2, 2>::operator+=(T s) {
    iadd(s);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>& Matrix<T, 2, 2>::operator+=(const Matrix& m) {
    iadd(m);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>& Matrix<T, 2, 2>::operator-=(T s) {
    isub(s);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>& Matrix<T, 2, 2>::operator-=(const Matrix& m) {
    isub(m);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>& Matrix<T, 2, 2>::operator*=(T s) {
    imul(s);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>& Matrix<T, 2, 2>::operator*=(const Matrix& m) {
    imul(m);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>& Matrix<T, 2, 2>::operator/=(T s) {
    idiv(s);
    return *this;
}

template <typename T>
bool Matrix<T, 2, 2>::operator==(const Matrix& m) const {
    return _elements[0] == m._elements[0] && _elements[1] == m._elements[1] &&
           _elements[2] == m._elements[2] && _elements[3] == m._elements[3];
}

template <typename T>
bool Matrix<T, 2, 2>::operator!=(const Matrix& m) const {
    return _elements[0] != m._elements[0] || _elements[1] != m._elements[1] ||
           _elements[2] != m._elements[2] || _elements[3] != m._elements[3];
}

// MARK: Getter operators
template <typename T>
T& Matrix<T, 2, 2>::operator[](size_t i) {
    return _elements[i];
}

template <typename T>
const T& Matrix<T, 2, 2>::operator[](size_t i) const {
    return _elements[i];
}

template <typename T>
T& Matrix<T, 2, 2>::operator()(size_t i, size_t j) {
    return _elements[2 * i + j];
}

template <typename T>
const T& Matrix<T, 2, 2>::operator()(size_t i, size_t j) const {
    return _elements[2 * i + j];
}

// MARK: Helpers
template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::makeZero() {
    return Matrix(0, 0, 0, 0);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::makeIdentity() {
    return Matrix(1, 0, 0, 1);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::makeScaleMatrix(T sx, T sy) {
    return Matrix(sx, 0, 0, sy);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::makeScaleMatrix(const Vector<T, 2>& s) {
    return makeScaleMatrix(s.x, s.y);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::makeRotationMatrix(const T& rad) {
    return Matrix(std::cos(rad), -std::sin(rad), std::sin(rad), std::cos(rad));
}

// MARK: Operator overloadings
template <typename T>
Matrix<T, 2, 2> operator-(const Matrix<T, 2, 2>& a) {
    return a.mul(-1);
}

template <typename T>
Matrix<T, 2, 2> operator+(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    return a.add(b);
}

template <typename T>
Matrix<T, 2, 2> operator+(const Matrix<T, 2, 2>& a, T b) {
    return a.add(b);
}

template <typename T>
Matrix<T, 2, 2> operator+(T a, const Matrix<T, 2, 2>& b) {
    return b.radd(a);
}

template <typename T>
Matrix<T, 2, 2> operator-(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    return a.sub(b);
}

template <typename T>
Matrix<T, 2, 2> operator-(const Matrix<T, 2, 2>& a, T b) {
    return a.sub(b);
}

template <typename T>
Matrix<T, 2, 2> operator-(T a, const Matrix<T, 2, 2>& b) {
    return b.rsub(a);
}

template <typename T>
Matrix<T, 2, 2> operator*(const Matrix<T, 2, 2>& a, T b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 2, 2> operator*(T a, const Matrix<T, 2, 2>& b) {
    return b.rmul(a);
}

template <typename T>
Vector<T, 3> operator*(const Matrix<T, 2, 2>& a, const Vector<T, 3>& b) {
    return a.mul(b);
}

template <typename T>
Vector<T, 2> operator*(const Matrix<T, 2, 2>& a, const Vector<T, 2>& b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 2, 2> operator*(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 2, 2> operator/(const Matrix<T, 2, 2>& a, T b) {
    return a.div(b);
}

template <typename T>
Matrix<T, 2, 2> operator/(T a, const Matrix<T, 2, 2>& b) {
    return b.rdiv(a);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_MATRIX2X2_INL_H_
