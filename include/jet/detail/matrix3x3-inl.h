// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_MATRIX3X3_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX3X3_INL_H_

#include <jet/math_utils.h>
#include <algorithm>
#include <cstring>
#include <utility>

namespace jet {

// MARK: CTOR/DTOR
template <typename T>
Matrix<T, 3, 3>::Matrix() {
    set(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

template <typename T>
Matrix<T, 3, 3>::Matrix(T s) {
    set(s);
}

template <typename T>
Matrix<T, 3, 3>::Matrix(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21,
                        T m22) {
    set(m00, m01, m02, m10, m11, m12, m20, m21, m22);
}

template <typename T>
template <typename U>
Matrix<T, 3, 3>::Matrix(
    const std::initializer_list<std::initializer_list<U>>& lst) {
    set(lst);
}

template <typename T>
Matrix<T, 3, 3>::Matrix(const Matrix& m) {
    set(m);
}

template <typename T>
Matrix<T, 3, 3>::Matrix(const T* arr) {
    set(arr);
}

// MARK: Basic setters
template <typename T>
void Matrix<T, 3, 3>::set(T s) {
    _elements[0] = _elements[3] = _elements[6] = _elements[1] = _elements[4] =
        _elements[7] = _elements[2] = _elements[5] = _elements[8] = s;
}

template <typename T>
void Matrix<T, 3, 3>::set(T m00, T m01, T m02, T m10, T m11, T m12, T m20,
                          T m21, T m22) {
    _elements[0] = m00;
    _elements[1] = m01;
    _elements[2] = m02;
    _elements[3] = m10;
    _elements[4] = m11;
    _elements[5] = m12;
    _elements[6] = m20;
    _elements[7] = m21;
    _elements[8] = m22;
}

template <typename T>
template <typename U>
void Matrix<T, 3, 3>::set(
    const std::initializer_list<std::initializer_list<U>>& lst) {
    size_t height = lst.size();
    size_t width = (height > 0) ? lst.begin()->size() : 0;
    JET_ASSERT(width == 3);
    JET_ASSERT(height == 3);

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
void Matrix<T, 3, 3>::set(const Matrix& m) {
    memcpy(_elements.data(), m._elements.data(), sizeof(T) * 9);
}

template <typename T>
void Matrix<T, 3, 3>::set(const T* arr) {
    memcpy(_elements.data(), arr, sizeof(T) * 9);
}

template <typename T>
void Matrix<T, 3, 3>::setDiagonal(T s) {
    _elements[0] = _elements[4] = _elements[8] = s;
}

template <typename T>
void Matrix<T, 3, 3>::setOffDiagonal(T s) {
    _elements[1] = _elements[2] = _elements[3] = _elements[5] = _elements[6] =
        _elements[7] = s;
}

template <typename T>
void Matrix<T, 3, 3>::setRow(size_t i, const Vector<T, 3>& row) {
    _elements[3 * i] = row.x;
    _elements[3 * i + 1] = row.y;
    _elements[3 * i + 2] = row.z;
}

template <typename T>
void Matrix<T, 3, 3>::setColumn(size_t j, const Vector<T, 3>& col) {
    _elements[j] = col.x;
    _elements[j + 3] = col.y;
    _elements[j + 6] = col.z;
}

// MARK: Basic getters
template <typename T>
bool Matrix<T, 3, 3>::isSimilar(const Matrix& m, double tol) const {
    return std::fabs(_elements[0] - m._elements[0]) < tol &&
           std::fabs(_elements[1] - m._elements[1]) < tol &&
           std::fabs(_elements[2] - m._elements[2]) < tol &&
           std::fabs(_elements[3] - m._elements[3]) < tol &&
           std::fabs(_elements[4] - m._elements[4]) < tol &&
           std::fabs(_elements[5] - m._elements[5]) < tol &&
           std::fabs(_elements[6] - m._elements[6]) < tol &&
           std::fabs(_elements[7] - m._elements[7]) < tol &&
           std::fabs(_elements[8] - m._elements[8]) < tol;
}

template <typename T>
bool Matrix<T, 3, 3>::isSquare() const {
    return true;
}

template <typename T>
size_t Matrix<T, 3, 3>::rows() const {
    return 3;
}

template <typename T>
size_t Matrix<T, 3, 3>::cols() const {
    return 3;
}

template <typename T>
T* Matrix<T, 3, 3>::data() {
    return _elements.data();
}

template <typename T>
const T* Matrix<T, 3, 3>::data() const {
    return _elements.data();
}

// MARK: Binary operator methods - new instance = this instance (+) input
template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::add(T s) const {
    return Matrix(_elements[0] + s, _elements[1] + s, _elements[2] + s,
                  _elements[3] + s, _elements[4] + s, _elements[5] + s,
                  _elements[6] + s, _elements[7] + s, _elements[8] + s);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::add(const Matrix& m) const {
    return Matrix(_elements[0] + m._elements[0], _elements[1] + m._elements[1],
                  _elements[2] + m._elements[2], _elements[3] + m._elements[3],
                  _elements[4] + m._elements[4], _elements[5] + m._elements[5],
                  _elements[6] + m._elements[6], _elements[7] + m._elements[7],
                  _elements[8] + m._elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::sub(T s) const {
    return Matrix(_elements[0] - s, _elements[1] - s, _elements[2] - s,
                  _elements[3] - s, _elements[4] - s, _elements[5] - s,
                  _elements[6] - s, _elements[7] - s, _elements[8] - s);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::sub(const Matrix& m) const {
    return Matrix(_elements[0] - m._elements[0], _elements[1] - m._elements[1],
                  _elements[2] - m._elements[2], _elements[3] - m._elements[3],
                  _elements[4] - m._elements[4], _elements[5] - m._elements[5],
                  _elements[6] - m._elements[6], _elements[7] - m._elements[7],
                  _elements[8] - m._elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::mul(T s) const {
    return Matrix(_elements[0] * s, _elements[1] * s, _elements[2] * s,
                  _elements[3] * s, _elements[4] * s, _elements[5] * s,
                  _elements[6] * s, _elements[7] * s, _elements[8] * s);
}

template <typename T>
Vector<T, 3> Matrix<T, 3, 3>::mul(const Vector<T, 3>& v) const {
    return Vector<T, 3>(
        _elements[0] * v.x + _elements[1] * v.y + _elements[2] * v.z,
        _elements[3] * v.x + _elements[4] * v.y + _elements[5] * v.z,
        _elements[6] * v.x + _elements[7] * v.y + _elements[8] * v.z);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::mul(const Matrix& m) const {
    return Matrix(
        _elements[0] * m._elements[0] + _elements[1] * m._elements[3] +
            _elements[2] * m._elements[6],
        _elements[0] * m._elements[1] + _elements[1] * m._elements[4] +
            _elements[2] * m._elements[7],
        _elements[0] * m._elements[2] + _elements[1] * m._elements[5] +
            _elements[2] * m._elements[8],

        _elements[3] * m._elements[0] + _elements[4] * m._elements[3] +
            _elements[5] * m._elements[6],
        _elements[3] * m._elements[1] + _elements[4] * m._elements[4] +
            _elements[5] * m._elements[7],
        _elements[3] * m._elements[2] + _elements[4] * m._elements[5] +
            _elements[5] * m._elements[8],

        _elements[6] * m._elements[0] + _elements[7] * m._elements[3] +
            _elements[8] * m._elements[6],
        _elements[6] * m._elements[1] + _elements[7] * m._elements[4] +
            _elements[8] * m._elements[7],
        _elements[6] * m._elements[2] + _elements[7] * m._elements[5] +
            _elements[8] * m._elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::div(T s) const {
    return Matrix(_elements[0] / s, _elements[1] / s, _elements[2] / s,
                  _elements[3] / s, _elements[4] / s, _elements[5] / s,
                  _elements[6] / s, _elements[7] / s, _elements[8] / s);
}

// MARK: Binary operator methods - new instance = input (+) this instance
template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::radd(T s) const {
    return Matrix(s + _elements[0], s + _elements[1], s + _elements[2],
                  s + _elements[3], s + _elements[4], s + _elements[5],
                  s + _elements[6], s + _elements[7], s + _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::radd(const Matrix& m) const {
    return Matrix(m._elements[0] + _elements[0], m._elements[1] + _elements[1],
                  m._elements[2] + _elements[2], m._elements[3] + _elements[3],
                  m._elements[4] + _elements[4], m._elements[5] + _elements[5],
                  m._elements[6] + _elements[6], m._elements[7] + _elements[7],
                  m._elements[8] + _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::rsub(T s) const {
    return Matrix(s - _elements[0], s - _elements[1], s - _elements[2],
                  s - _elements[3], s - _elements[4], s - _elements[5],
                  s - _elements[6], s - _elements[7], s - _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::rsub(const Matrix& m) const {
    return Matrix(m._elements[0] - _elements[0], m._elements[1] - _elements[1],
                  m._elements[2] - _elements[2], m._elements[3] - _elements[3],
                  m._elements[4] - _elements[4], m._elements[5] - _elements[5],
                  m._elements[6] - _elements[6], m._elements[7] - _elements[7],
                  m._elements[8] - _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::rmul(T s) const {
    return Matrix(s * _elements[0], s * _elements[1], s * _elements[2],
                  s * _elements[3], s * _elements[4], s * _elements[5],
                  s * _elements[6], s * _elements[7], s * _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::rmul(const Matrix& m) const {
    return m.mul(*this);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::rdiv(T s) const {
    return Matrix(s / _elements[0], s / _elements[1], s / _elements[2],
                  s / _elements[3], s / _elements[4], s / _elements[5],
                  s / _elements[6], s / _elements[7], s / _elements[8]);
}

// MARK: Augmented operator methods - this instance (+)= input
template <typename T>
void Matrix<T, 3, 3>::iadd(T s) {
    _elements[0] += s;
    _elements[1] += s;
    _elements[2] += s;
    _elements[3] += s;
    _elements[4] += s;
    _elements[5] += s;
    _elements[6] += s;
    _elements[7] += s;
    _elements[8] += s;
}

template <typename T>
void Matrix<T, 3, 3>::iadd(const Matrix& m) {
    _elements[0] += m._elements[0];
    _elements[1] += m._elements[1];
    _elements[2] += m._elements[2];
    _elements[3] += m._elements[3];
    _elements[4] += m._elements[4];
    _elements[5] += m._elements[5];
    _elements[6] += m._elements[6];
    _elements[7] += m._elements[7];
    _elements[8] += m._elements[8];
}

template <typename T>
void Matrix<T, 3, 3>::isub(T s) {
    _elements[0] -= s;
    _elements[1] -= s;
    _elements[2] -= s;
    _elements[3] -= s;
    _elements[4] -= s;
    _elements[5] -= s;
    _elements[6] -= s;
    _elements[7] -= s;
    _elements[8] -= s;
}

template <typename T>
void Matrix<T, 3, 3>::isub(const Matrix& m) {
    _elements[0] -= m._elements[0];
    _elements[1] -= m._elements[1];
    _elements[2] -= m._elements[2];
    _elements[3] -= m._elements[3];
    _elements[4] -= m._elements[4];
    _elements[5] -= m._elements[5];
    _elements[6] -= m._elements[6];
    _elements[7] -= m._elements[7];
    _elements[8] -= m._elements[8];
}

template <typename T>
void Matrix<T, 3, 3>::imul(T s) {
    _elements[0] *= s;
    _elements[1] *= s;
    _elements[2] *= s;
    _elements[3] *= s;
    _elements[4] *= s;
    _elements[5] *= s;
    _elements[6] *= s;
    _elements[7] *= s;
    _elements[8] *= s;
}

template <typename T>
void Matrix<T, 3, 3>::imul(const Matrix& m) {
    set(mul(m));
}

template <typename T>
void Matrix<T, 3, 3>::idiv(T s) {
    _elements[0] /= s;
    _elements[1] /= s;
    _elements[2] /= s;
    _elements[3] /= s;
    _elements[4] /= s;
    _elements[5] /= s;
    _elements[6] /= s;
    _elements[7] /= s;
    _elements[8] /= s;
}

// MARK: Modifiers
template <typename T>
void Matrix<T, 3, 3>::transpose() {
    std::swap(_elements[1], _elements[3]);
    std::swap(_elements[2], _elements[6]);
    std::swap(_elements[5], _elements[7]);
}

template <typename T>
void Matrix<T, 3, 3>::invert() {
    T d = determinant();

    Matrix m;
    m._elements[0] = _elements[4] * _elements[8] - _elements[5] * _elements[7];
    m._elements[1] = _elements[2] * _elements[7] - _elements[1] * _elements[8];
    m._elements[2] = _elements[1] * _elements[5] - _elements[2] * _elements[4];
    m._elements[3] = _elements[5] * _elements[6] - _elements[3] * _elements[8];
    m._elements[4] = _elements[0] * _elements[8] - _elements[2] * _elements[6];
    m._elements[5] = _elements[2] * _elements[3] - _elements[0] * _elements[5];
    m._elements[6] = _elements[3] * _elements[7] - _elements[4] * _elements[6];
    m._elements[7] = _elements[1] * _elements[6] - _elements[0] * _elements[7];
    m._elements[8] = _elements[0] * _elements[4] - _elements[1] * _elements[3];
    m.idiv(d);

    set(m);
}

// MARK: Complex getters
template <typename T>
T Matrix<T, 3, 3>::sum() const {
    T s = 0;
    for (int i = 0; i < 9; ++i) {
        s += _elements[i];
    }
    return s;
}

template <typename T>
T Matrix<T, 3, 3>::avg() const {
    return sum() / 9;
}

template <typename T>
T Matrix<T, 3, 3>::min() const {
    return minn(data(), 9);
}

template <typename T>
T Matrix<T, 3, 3>::max() const {
    return maxn(data(), 9);
}

template <typename T>
T Matrix<T, 3, 3>::absmin() const {
    return absminn(data(), 9);
}

template <typename T>
T Matrix<T, 3, 3>::absmax() const {
    return absmaxn(data(), 9);
}

template <typename T>
T Matrix<T, 3, 3>::trace() const {
    return _elements[0] + _elements[4] + _elements[8];
}

template <typename T>
T Matrix<T, 3, 3>::determinant() const {
    return _elements[0] * _elements[4] * _elements[8] -
           _elements[0] * _elements[5] * _elements[7] +
           _elements[1] * _elements[5] * _elements[6] -
           _elements[1] * _elements[3] * _elements[8] +
           _elements[2] * _elements[3] * _elements[7] -
           _elements[2] * _elements[4] * _elements[6];
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::diagonal() const {
    return Matrix(_elements[0], 0, 0, 0, _elements[4], 0, 0, 0, _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::offDiagonal() const {
    return Matrix(0, _elements[1], _elements[2], _elements[3], 0, _elements[5],
                  _elements[6], _elements[7], 0);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::strictLowerTri() const {
    return Matrix(0, 0, 0, _elements[3], 0, 0, _elements[6], _elements[7], 0);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::strictUpperTri() const {
    return Matrix(0, _elements[1], _elements[2], 0, 0, _elements[5], 0, 0, 0);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::lowerTri() const {
    return Matrix(_elements[0], 0, 0, _elements[3], _elements[4], 0,
                  _elements[6], _elements[7], _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::upperTri() const {
    return Matrix(_elements[0], _elements[1], _elements[2], 0, _elements[4],
                  _elements[5], 0, 0, _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::transposed() const {
    return Matrix(_elements[0], _elements[3], _elements[6], _elements[1],
                  _elements[4], _elements[7], _elements[2], _elements[5],
                  _elements[8]);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::inverse() const {
    Matrix m(*this);
    m.invert();
    return m;
}

template <typename T>
T Matrix<T, 3, 3>::frobeniusNorm() const {
    return std::sqrt(_elements[0] * _elements[0] + _elements[1] * _elements[1] +
                     _elements[2] * _elements[2] + _elements[3] * _elements[3] +
                     _elements[4] * _elements[4] + _elements[5] * _elements[5] +
                     _elements[6] * _elements[6] + _elements[7] * _elements[7] +
                     _elements[8] * _elements[8]);
}

template <typename T>
template <typename U>
Matrix<U, 3, 3> Matrix<T, 3, 3>::castTo() const {
    return Matrix<U, 3, 3>(
        static_cast<U>(_elements[0]), static_cast<U>(_elements[1]),
        static_cast<U>(_elements[2]), static_cast<U>(_elements[3]),
        static_cast<U>(_elements[4]), static_cast<U>(_elements[5]),
        static_cast<U>(_elements[6]), static_cast<U>(_elements[7]),
        static_cast<U>(_elements[8]));
}

// MARK: Setter operators
template <typename T>
Matrix<T, 3, 3>& Matrix<T, 3, 3>::operator=(const Matrix& m) {
    set(m);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>& Matrix<T, 3, 3>::operator+=(T s) {
    iadd(s);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>& Matrix<T, 3, 3>::operator+=(const Matrix& m) {
    iadd(m);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>& Matrix<T, 3, 3>::operator-=(T s) {
    isub(s);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>& Matrix<T, 3, 3>::operator-=(const Matrix& m) {
    isub(m);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>& Matrix<T, 3, 3>::operator*=(T s) {
    imul(s);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>& Matrix<T, 3, 3>::operator*=(const Matrix& m) {
    imul(m);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>& Matrix<T, 3, 3>::operator/=(T s) {
    idiv(s);
    return *this;
}

template <typename T>
bool Matrix<T, 3, 3>::operator==(const Matrix& m) const {
    return _elements[0] == m._elements[0] && _elements[1] == m._elements[1] &&
           _elements[2] == m._elements[2] && _elements[3] == m._elements[3] &&
           _elements[4] == m._elements[4] && _elements[5] == m._elements[5] &&
           _elements[6] == m._elements[6] && _elements[7] == m._elements[7] &&
           _elements[8] == m._elements[8];
}

template <typename T>
bool Matrix<T, 3, 3>::operator!=(const Matrix& m) const {
    return _elements[0] != m._elements[0] || _elements[1] != m._elements[1] ||
           _elements[2] != m._elements[2] || _elements[3] != m._elements[3] ||
           _elements[4] != m._elements[4] || _elements[5] != m._elements[5] ||
           _elements[6] != m._elements[6] || _elements[7] != m._elements[7] ||
           _elements[8] != m._elements[8];
}

// MARK: Getter operators
template <typename T>
T& Matrix<T, 3, 3>::operator[](size_t i) {
    return _elements[i];
}

template <typename T>
const T& Matrix<T, 3, 3>::operator[](size_t i) const {
    return _elements[i];
}

template <typename T>
T& Matrix<T, 3, 3>::operator()(size_t i, size_t j) {
    return _elements[3 * i + j];
}

template <typename T>
const T& Matrix<T, 3, 3>::operator()(size_t i, size_t j) const {
    return _elements[3 * i + j];
}

// MARK: Helpers
template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::makeZero() {
    return Matrix(0, 0, 0, 0, 0, 0, 0, 0, 0);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::makeIdentity() {
    return Matrix(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::makeScaleMatrix(T sx, T sy, T sz) {
    return Matrix(sx, 0, 0, 0, sy, 0, 0, 0, sz);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::makeScaleMatrix(const Vector<T, 3>& s) {
    return makeScaleMatrix(s.x, s.y, s.z);
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 3, 3>::makeRotationMatrix(const Vector<T, 3>& axis,
                                                    T rad) {
    return Matrix(
        1 + (1 - std::cos(rad)) * (axis.x * axis.x - 1),
        -axis.z * std::sin(rad) + (1 - std::cos(rad)) * axis.x * axis.y,
        axis.y * std::sin(rad) + (1 - std::cos(rad)) * axis.x * axis.z,

        axis.z * std::sin(rad) + (1 - std::cos(rad)) * axis.x * axis.y,
        1 + (1 - std::cos(rad)) * (axis.y * axis.y - 1),
        -axis.x * std::sin(rad) + (1 - std::cos(rad)) * axis.y * axis.z,

        -axis.y * std::sin(rad) + (1 - std::cos(rad)) * axis.x * axis.z,
        axis.x * std::sin(rad) + (1 - std::cos(rad)) * axis.y * axis.z,
        1 + (1 - std::cos(rad)) * (axis.z * axis.z - 1));
}

// MARK: Operator overloadings
template <typename T>
Matrix<T, 3, 3> operator-(const Matrix<T, 3, 3>& a) {
    return a.mul(-1);
}

template <typename T>
Matrix<T, 3, 3> operator+(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b) {
    return a.add(b);
}

template <typename T>
Matrix<T, 3, 3> operator+(const Matrix<T, 3, 3>& a, T b) {
    return a.add(b);
}

template <typename T>
Matrix<T, 3, 3> operator+(T a, const Matrix<T, 3, 3>& b) {
    return b.radd(a);
}

template <typename T>
Matrix<T, 3, 3> operator-(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b) {
    return a.sub(b);
}

template <typename T>
Matrix<T, 3, 3> operator-(const Matrix<T, 3, 3>& a, T b) {
    return a.sub(b);
}

template <typename T>
Matrix<T, 3, 3> operator-(T a, const Matrix<T, 3, 3>& b) {
    return b.rsub(a);
}

template <typename T>
Matrix<T, 3, 3> operator*(const Matrix<T, 3, 3>& a, T b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 3, 3> operator*(T a, const Matrix<T, 3, 3>& b) {
    return b.rmul(a);
}

template <typename T>
Vector<T, 3> operator*(const Matrix<T, 3, 3>& a, const Vector<T, 3>& b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 3, 3> operator*(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 3, 3> operator/(const Matrix<T, 3, 3>& a, T b) {
    return a.div(b);
}

template <typename T>
Matrix<T, 3, 3> operator/(T a, const Matrix<T, 3, 3>& b) {
    return b.rdiv(a);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_MATRIX3X3_INL_H_
