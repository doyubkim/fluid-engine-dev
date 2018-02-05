// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_MATRIX4X4_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX4X4_INL_H_

#include <jet/math_utils.h>
#include <algorithm>
#include <cstring>
#include <utility>

namespace jet {

// MARK: CTOR/DTOR
template <typename T>
Matrix<T, 4, 4>::Matrix() {
    set(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
}

template <typename T>
Matrix<T, 4, 4>::Matrix(T s) {
    set(s);
}

template <typename T>
Matrix<T, 4, 4>::Matrix(
    T m00, T m01, T m02,
    T m10, T m11, T m12,
    T m20, T m21, T m22) {
    set(m00, m01, m02, m10, m11, m12, m20, m21, m22);
}

template <typename T>
Matrix<T, 4, 4>::Matrix(
    T m00, T m01, T m02, T m03,
    T m10, T m11, T m12, T m13,
    T m20, T m21, T m22, T m23,
    T m30, T m31, T m32, T m33) {
    set(m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33);
}

template <typename T>
template <typename U>
Matrix<T, 4, 4>::Matrix(
    const std::initializer_list<std::initializer_list<U>>& lst) {
    set(lst);
}

template <typename T>
Matrix<T, 4, 4>::Matrix(const Matrix<T, 3, 3>& m33) {
    set(m33);
}

template <typename T>
Matrix<T, 4, 4>::Matrix(const Matrix& m) {
    set(m);
}

template <typename T>
Matrix<T, 4, 4>::Matrix(const T* arr) {
    set(arr);
}


// MARK: Basic setters
template <typename T>
void Matrix<T, 4, 4>::set(T s) {
    _elements[0] = _elements[4] = _elements[8] = _elements[12] =
    _elements[1] = _elements[5] = _elements[9] = _elements[13] =
    _elements[2] = _elements[6] = _elements[10] = _elements[14] =
    _elements[3] = _elements[7] = _elements[11] = _elements[15] = s;
}

template <typename T>
void Matrix<T, 4, 4>::set(
    T m00, T m01, T m02,
    T m10, T m11, T m12,
    T m20, T m21, T m22) {
    _elements[0] = m00; _elements[1] = m01; _elements[2] = m02;  _elements[3] = 0;
    _elements[4] = m10; _elements[5] = m11; _elements[6] = m12;  _elements[7] = 0;
    _elements[8] = m20; _elements[9] = m21; _elements[10] = m22; _elements[11] = 0;
    _elements[12] = 0;  _elements[13] = 0;  _elements[14] = 0;   _elements[15] = 1;
}

template <typename T>
void Matrix<T, 4, 4>::set(
    T m00, T m01, T m02, T m03,
    T m10, T m11, T m12, T m13,
    T m20, T m21, T m22, T m23,
    T m30, T m31, T m32, T m33) {
    _elements[0] = m00;
    _elements[1] = m01;
    _elements[2] = m02;
    _elements[3] = m03;
    _elements[4] = m10;
    _elements[5] = m11;
    _elements[6] = m12;
    _elements[7] = m13;
    _elements[8] = m20;
    _elements[9] = m21;
    _elements[10] = m22;
    _elements[11] = m23;
    _elements[12] = m30;
    _elements[13] = m31;
    _elements[14] = m32;
    _elements[15] = m33;
}

template <typename T>
template <typename U>
void Matrix<T, 4, 4>::set(
    const std::initializer_list<std::initializer_list<U>>& lst) {
    size_t height = lst.size();
    size_t width = (height > 0) ? lst.begin()->size() : 0;
    JET_ASSERT(width == 4);
    JET_ASSERT(height == 4);

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
void Matrix<T, 4, 4>::set(const Matrix<T, 3, 3>& m33) {
    set(m33[0], m33[1], m33[2], 0,
        m33[3], m33[4], m33[5], 0,
        m33[6], m33[7], m33[8], 0,
        0,      0,      0,      1);
}

template <typename T>
void Matrix<T, 4, 4>::set(const Matrix& m) {
    _elements = m._elements;
}

template <typename T>
void Matrix<T, 4, 4>::set(const T* arr) {
    memcpy(_elements.data(), arr, sizeof(T) * 16);
}

template <typename T>
void Matrix<T, 4, 4>::setDiagonal(T s) {
    _elements[0] = _elements[5] = _elements[10] = _elements[15] = s;
}

template <typename T>
void Matrix<T, 4, 4>::setOffDiagonal(T s) {
    _elements[1] = _elements[2] = _elements[3] = _elements[4]
    = _elements[6] = _elements[7] = _elements[8] = _elements[9]
    = _elements[11] = _elements[12] = _elements[13] = _elements[14] = s;
}

template <typename T>
void Matrix<T, 4, 4>::setRow(size_t i, const Vector<T, 4>& row) {
    _elements[4 * i] = row.x;
    _elements[4 * i + 1] = row.y;
    _elements[4 * i + 2] = row.z;
    _elements[4 * i + 3] = row.w;
}

template <typename T>
void Matrix<T, 4, 4>::setColumn(size_t j, const Vector<T, 4>& col) {
    _elements[j] = col.x;
    _elements[j + 4] = col.y;
    _elements[j + 8] = col.z;
    _elements[j + 12] = col.w;
}


// MARK: Basic getters
template <typename T>
bool Matrix<T, 4, 4>::isSimilar(const Matrix& m, double tol) const {
    return
        std::fabs(_elements[0] - m._elements[0]) < tol
     && std::fabs(_elements[1] - m._elements[1]) < tol
     && std::fabs(_elements[2] - m._elements[2]) < tol
     && std::fabs(_elements[3] - m._elements[3]) < tol
     && std::fabs(_elements[4] - m._elements[4]) < tol
     && std::fabs(_elements[5] - m._elements[5]) < tol
     && std::fabs(_elements[6] - m._elements[6]) < tol
     && std::fabs(_elements[7] - m._elements[7]) < tol
     && std::fabs(_elements[8] - m._elements[8]) < tol
     && std::fabs(_elements[9] - m._elements[9]) < tol
     && std::fabs(_elements[10] - m._elements[10]) < tol
     && std::fabs(_elements[11] - m._elements[11]) < tol
     && std::fabs(_elements[12] - m._elements[12]) < tol
     && std::fabs(_elements[13] - m._elements[13]) < tol
     && std::fabs(_elements[14] - m._elements[14]) < tol
     && std::fabs(_elements[15] - m._elements[15]) < tol;
}

template <typename T>
bool Matrix<T, 4, 4>::isSquare() const {
    return true;
}

template <typename T>
size_t Matrix<T, 4, 4>::rows() const {
    return 4;
}

template <typename T>
size_t Matrix<T, 4, 4>::cols() const {
    return 4;
}

template <typename T>
T* Matrix<T, 4, 4>::data() {
    return _elements.data();
}

template <typename T>
const T* Matrix<T, 4, 4>::data() const {
    return _elements.data();
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 4, 4>::matrix3() const {
    return Matrix<T, 3, 3>(
        _elements[0], _elements[1], _elements[2],
        _elements[4], _elements[5], _elements[6],
        _elements[8], _elements[9], _elements[10]);
}


// MARK: Binary operator methods - new instance = this instance (+) input
template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::add(T s) const {
    return Matrix(
        _elements[0] + s,
        _elements[1] + s,
        _elements[2] + s,
        _elements[3] + s,
        _elements[4] + s,
        _elements[5] + s,
        _elements[6] + s,
        _elements[7] + s,
        _elements[8] + s,
        _elements[9] + s,
        _elements[10] + s,
        _elements[11] + s,
        _elements[12] + s,
        _elements[13] + s,
        _elements[14] + s,
        _elements[15] + s);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::add(const Matrix& m) const {
    return Matrix(
        _elements[0] + m._elements[0],
        _elements[1] + m._elements[1],
        _elements[2] + m._elements[2],
        _elements[3] + m._elements[3],
        _elements[4] + m._elements[4],
        _elements[5] + m._elements[5],
        _elements[6] + m._elements[6],
        _elements[7] + m._elements[7],
        _elements[8] + m._elements[8],
        _elements[9] + m._elements[9],
        _elements[10] + m._elements[10],
        _elements[11] + m._elements[11],
        _elements[12] + m._elements[12],
        _elements[13] + m._elements[13],
        _elements[14] + m._elements[14],
        _elements[15] + m._elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::sub(T s) const {
    return Matrix(
        _elements[0] - s, _elements[1] - s, _elements[2] - s, _elements[3] - s,
        _elements[4] - s, _elements[5] - s, _elements[6] - s, _elements[7] - s,
        _elements[8] - s, _elements[9] - s, _elements[10] - s, _elements[11] - s,
        _elements[12] - s, _elements[13] - s, _elements[14] - s, _elements[15] - s);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::sub(const Matrix& m) const {
    return Matrix(
        _elements[0] - m._elements[0],
        _elements[1] - m._elements[1],
        _elements[2] - m._elements[2],
        _elements[3] - m._elements[3],
        _elements[4] - m._elements[4],
        _elements[5] - m._elements[5],
        _elements[6] - m._elements[6],
        _elements[7] - m._elements[7],
        _elements[8] - m._elements[8],
        _elements[9] - m._elements[9],
        _elements[10] - m._elements[10],
        _elements[11] - m._elements[11],
        _elements[12] - m._elements[12],
        _elements[13] - m._elements[13],
        _elements[14] - m._elements[14],
        _elements[15] - m._elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::mul(T s) const {
    return Matrix(
        _elements[0] * s, _elements[1] * s, _elements[2] * s, _elements[3] * s,
        _elements[4] * s, _elements[5] * s, _elements[6] * s, _elements[7] * s,
        _elements[8] * s, _elements[9] * s, _elements[10] * s, _elements[11] * s,
        _elements[12] * s, _elements[13] * s, _elements[14] * s, _elements[15] * s);
}

template <typename T>
Vector<T, 4> Matrix<T, 4, 4>::mul(const Vector<T, 4>& v) const {
    return Vector<T, 4>(
        _elements[0] * v.x + _elements[1] * v.y + _elements[2] * v.z + _elements[3] * v.w,
        _elements[4] * v.x + _elements[5] * v.y + _elements[6] * v.z + _elements[7] * v.w,
        _elements[8] * v.x + _elements[9] * v.y + _elements[10] * v.z + _elements[11] * v.w,
        _elements[12] * v.x + _elements[13] * v.y + _elements[14] * v.z + _elements[15] * v.w);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::mul(const Matrix& m) const {
    return Matrix(
        _elements[0] * m._elements[0] + _elements[1] * m._elements[4] + _elements[2] * m._elements[8] + _elements[3] * m._elements[12],
        _elements[0] * m._elements[1] + _elements[1] * m._elements[5] + _elements[2] * m._elements[9] + _elements[3] * m._elements[13],
        _elements[0] * m._elements[2] + _elements[1] * m._elements[6] + _elements[2] * m._elements[10] + _elements[3] * m._elements[14],
        _elements[0] * m._elements[3] + _elements[1] * m._elements[7] + _elements[2] * m._elements[11] + _elements[3] * m._elements[15],

        _elements[4] * m._elements[0] + _elements[5] * m._elements[4] + _elements[6] * m._elements[8] + _elements[7] * m._elements[12],
        _elements[4] * m._elements[1] + _elements[5] * m._elements[5] + _elements[6] * m._elements[9] + _elements[7] * m._elements[13],
        _elements[4] * m._elements[2] + _elements[5] * m._elements[6] + _elements[6] * m._elements[10] + _elements[7] * m._elements[14],
        _elements[4] * m._elements[3] + _elements[5] * m._elements[7] + _elements[6] * m._elements[11] + _elements[7] * m._elements[15],

        _elements[8] * m._elements[0] + _elements[9] * m._elements[4] + _elements[10] * m._elements[8] + _elements[11] * m._elements[12],
        _elements[8] * m._elements[1] + _elements[9] * m._elements[5] + _elements[10] * m._elements[9] + _elements[11] * m._elements[13],
        _elements[8] * m._elements[2] + _elements[9] * m._elements[6] + _elements[10] * m._elements[10] + _elements[11] * m._elements[14],
        _elements[8] * m._elements[3] + _elements[9] * m._elements[7] + _elements[10] * m._elements[11] + _elements[11] * m._elements[15],

        _elements[12] * m._elements[0] + _elements[13] * m._elements[4] + _elements[14] * m._elements[8] + _elements[15] * m._elements[12],
        _elements[12] * m._elements[1] + _elements[13] * m._elements[5] + _elements[14] * m._elements[9] + _elements[15] * m._elements[13],
        _elements[12] * m._elements[2] + _elements[13] * m._elements[6] + _elements[14] * m._elements[10] + _elements[15] * m._elements[14],
        _elements[12] * m._elements[3] + _elements[13] * m._elements[7] + _elements[14] * m._elements[11] + _elements[15] * m._elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::div(T s) const {
    return Matrix(
        _elements[0] / s, _elements[1] / s, _elements[2] / s, _elements[3] / s,
        _elements[4] / s, _elements[5] / s, _elements[6] / s, _elements[7] / s,
        _elements[8] / s, _elements[9] / s, _elements[10] / s, _elements[11] / s,
        _elements[12] / s, _elements[13] / s, _elements[14] / s, _elements[15] / s);
}


// MARK: Binary operator methods - new instance = input (+) this instance
template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::radd(T s) const {
    return Matrix(
        s + _elements[0], s + _elements[1], s + _elements[2], s + _elements[3],
        s + _elements[4], s + _elements[5], s + _elements[6], s + _elements[7],
        s + _elements[8], s + _elements[9], s + _elements[10], s + _elements[11],
        s + _elements[12], s + _elements[13], s + _elements[14], s + _elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::radd(const Matrix& m) const {
    return Matrix(
        m._elements[0] + _elements[0], m._elements[1] + _elements[1], m._elements[2] + _elements[2], m._elements[3] + _elements[3],
        m._elements[4] + _elements[4], m._elements[5] + _elements[5], m._elements[6] + _elements[6], m._elements[7] + _elements[7],
        m._elements[8] + _elements[8], m._elements[9] + _elements[9], m._elements[10] + _elements[10], m._elements[11] + _elements[11],
        m._elements[12] + _elements[12], m._elements[13] + _elements[13], m._elements[14] + _elements[14], m._elements[15] + _elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rsub(T s) const {
    return Matrix(
        s - _elements[0], s - _elements[1], s - _elements[2], s - _elements[3],
        s - _elements[4], s - _elements[5], s - _elements[6], s - _elements[7],
        s - _elements[8], s - _elements[9], s - _elements[10], s - _elements[11],
        s - _elements[12], s - _elements[13], s - _elements[14], s - _elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rsub(const Matrix& m) const {
    return Matrix(
        m._elements[0] - _elements[0], m._elements[1] - _elements[1], m._elements[2] - _elements[2], m._elements[3] - _elements[3],
        m._elements[4] - _elements[4], m._elements[5] - _elements[5], m._elements[6] - _elements[6], m._elements[7] - _elements[7],
        m._elements[8] - _elements[8], m._elements[9] - _elements[9], m._elements[10] - _elements[10], m._elements[11] - _elements[11],
        m._elements[12] - _elements[12], m._elements[13] - _elements[13], m._elements[14] - _elements[14], m._elements[15] - _elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rmul(T s) const {
    return Matrix(
        s*_elements[0], s*_elements[1], s*_elements[2], s*_elements[3],
        s*_elements[4], s*_elements[5], s*_elements[6], s*_elements[7],
        s*_elements[8], s*_elements[9], s*_elements[10], s*_elements[11],
        s*_elements[12], s*_elements[13], s*_elements[14], s*_elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rmul(const Matrix& m) const {
    return m.mul(*this);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rdiv(T s) const {
    return Matrix(
        s / _elements[0], s / _elements[1], s / _elements[2], s / _elements[3],
        s / _elements[4], s / _elements[5], s / _elements[6], s / _elements[7],
        s / _elements[8], s / _elements[9], s / _elements[10], s / _elements[11],
        s / _elements[12], s / _elements[13], s / _elements[14], s / _elements[15]);
}

// MARK: Augmented operator methods - this instance (+)= input
template <typename T>
void Matrix<T, 4, 4>::iadd(T s) {
    _elements[0] += s; _elements[1] += s; _elements[2] += s; _elements[3] += s;
    _elements[4] += s; _elements[5] += s; _elements[6] += s; _elements[7] += s;
    _elements[8] += s; _elements[9] += s; _elements[10] += s; _elements[11] += s;
    _elements[12] += s; _elements[13] += s; _elements[14] += s; _elements[15] += s;
}

template <typename T>
void Matrix<T, 4, 4>::iadd(const Matrix& m) {
    _elements[0] += m._elements[0]; _elements[1] += m._elements[1]; _elements[2] += m._elements[2]; _elements[3] += m._elements[3];
    _elements[4] += m._elements[4]; _elements[5] += m._elements[5]; _elements[6] += m._elements[6]; _elements[7] += m._elements[7];
    _elements[8] += m._elements[8]; _elements[9] += m._elements[9]; _elements[10] += m._elements[10]; _elements[11] += m._elements[11];
    _elements[12] += m._elements[12]; _elements[13] += m._elements[13]; _elements[14] += m._elements[14]; _elements[15] += m._elements[15];
}

template <typename T>
void Matrix<T, 4, 4>::isub(T s) {
    _elements[0] -= s; _elements[1] -= s; _elements[2] -= s; _elements[3] -= s;
    _elements[4] -= s; _elements[5] -= s; _elements[6] -= s; _elements[7] -= s;
    _elements[8] -= s; _elements[9] -= s; _elements[10] -= s; _elements[11] -= s;
    _elements[12] -= s; _elements[13] -= s; _elements[14] -= s; _elements[15] -= s;
}

template <typename T>
void Matrix<T, 4, 4>::isub(const Matrix& m) {
    _elements[0] -= m._elements[0]; _elements[1] -= m._elements[1]; _elements[2] -= m._elements[2]; _elements[3] -= m._elements[3];
    _elements[4] -= m._elements[4]; _elements[5] -= m._elements[5]; _elements[6] -= m._elements[6]; _elements[7] -= m._elements[7];
    _elements[8] -= m._elements[8]; _elements[9] -= m._elements[9]; _elements[10] -= m._elements[10]; _elements[11] -= m._elements[11];
    _elements[12] -= m._elements[12]; _elements[13] -= m._elements[13]; _elements[14] -= m._elements[14]; _elements[15] -= m._elements[15];
}

template <typename T>
void Matrix<T, 4, 4>::imul(T s) {
    _elements[0] *= s; _elements[1] *= s; _elements[2] *= s; _elements[3] *= s;
    _elements[4] *= s; _elements[5] *= s; _elements[6] *= s; _elements[7] *= s;
    _elements[8] *= s; _elements[9] *= s; _elements[10] *= s; _elements[11] *= s;
    _elements[12] *= s; _elements[13] *= s; _elements[14] *= s; _elements[15] *= s;
}

template <typename T>
void Matrix<T, 4, 4>::imul(const Matrix& m) {
    set(mul(m));
}

template <typename T>
void Matrix<T, 4, 4>::idiv(T s) {
    _elements[0] /= s; _elements[1] /= s; _elements[2] /= s; _elements[3] /= s;
    _elements[4] /= s; _elements[5] /= s; _elements[6] /= s; _elements[7] /= s;
    _elements[8] /= s; _elements[9] /= s; _elements[10] /= s; _elements[11] /= s;
    _elements[12] /= s; _elements[13] /= s; _elements[14] /= s; _elements[15] /= s;
}


// MARK: Modifiers
template <typename T>
void Matrix<T, 4, 4>::transpose() {
    std::swap(_elements[1], _elements[4]);
    std::swap(_elements[2], _elements[8]);
    std::swap(_elements[3], _elements[12]);
    std::swap(_elements[6], _elements[9]);
    std::swap(_elements[7], _elements[13]);
    std::swap(_elements[11], _elements[14]);
}

template <typename T>
void Matrix<T, 4, 4>::invert() {
    T d = determinant();
    Matrix m;
    m._elements[0] = _elements[5] * _elements[10] * _elements[15] + _elements[6] * _elements[11] * _elements[13] + _elements[7] * _elements[9] * _elements[14] - _elements[5] * _elements[11] * _elements[14] - _elements[6] * _elements[9] * _elements[15] - _elements[7] * _elements[10] * _elements[13];
    m._elements[1] = _elements[1] * _elements[11] * _elements[14] + _elements[2] * _elements[9] * _elements[15] + _elements[3] * _elements[10] * _elements[13] - _elements[1] * _elements[10] * _elements[15] - _elements[2] * _elements[11] * _elements[13] - _elements[3] * _elements[9] * _elements[14];
    m._elements[2] = _elements[1] * _elements[6] * _elements[15] + _elements[2] * _elements[7] * _elements[13] + _elements[3] * _elements[5] * _elements[14] - _elements[1] * _elements[7] * _elements[14] - _elements[2] * _elements[5] * _elements[15] - _elements[3] * _elements[6] * _elements[13];
    m._elements[3] = _elements[1] * _elements[7] * _elements[10] + _elements[2] * _elements[5] * _elements[11] + _elements[3] * _elements[6] * _elements[9] - _elements[1] * _elements[6] * _elements[11] - _elements[2] * _elements[7] * _elements[9] - _elements[3] * _elements[5] * _elements[10];
    m._elements[4] = _elements[4] * _elements[11] * _elements[14] + _elements[6] * _elements[8] * _elements[15] + _elements[7] * _elements[10] * _elements[12] - _elements[4] * _elements[10] * _elements[15] - _elements[6] * _elements[11] * _elements[12] - _elements[7] * _elements[8] * _elements[14];
    m._elements[5] = _elements[0] * _elements[10] * _elements[15] + _elements[2] * _elements[11] * _elements[12] + _elements[3] * _elements[8] * _elements[14] - _elements[0] * _elements[11] * _elements[14] - _elements[2] * _elements[8] * _elements[15] - _elements[3] * _elements[10] * _elements[12];
    m._elements[6] = _elements[0] * _elements[7] * _elements[14] + _elements[2] * _elements[4] * _elements[15] + _elements[3] * _elements[6] * _elements[12] - _elements[0] * _elements[6] * _elements[15] - _elements[2] * _elements[7] * _elements[12] - _elements[3] * _elements[4] * _elements[14];
    m._elements[7] = _elements[0] * _elements[6] * _elements[11] + _elements[2] * _elements[7] * _elements[8] + _elements[3] * _elements[4] * _elements[10] - _elements[0] * _elements[7] * _elements[10] - _elements[2] * _elements[4] * _elements[11] - _elements[3] * _elements[6] * _elements[8];
    m._elements[8] = _elements[4] * _elements[9] * _elements[15] + _elements[5] * _elements[11] * _elements[12] + _elements[7] * _elements[8] * _elements[13] - _elements[4] * _elements[11] * _elements[13] - _elements[5] * _elements[8] * _elements[15] - _elements[7] * _elements[9] * _elements[12];
    m._elements[9] = _elements[0] * _elements[11] * _elements[13] + _elements[1] * _elements[8] * _elements[15] + _elements[3] * _elements[9] * _elements[12] - _elements[0] * _elements[9] * _elements[15] - _elements[1] * _elements[11] * _elements[12] - _elements[3] * _elements[8] * _elements[13];
    m._elements[10] = _elements[0] * _elements[5] * _elements[15] + _elements[1] * _elements[7] * _elements[12] + _elements[3] * _elements[4] * _elements[13] - _elements[0] * _elements[7] * _elements[13] - _elements[1] * _elements[4] * _elements[15] - _elements[3] * _elements[5] * _elements[12];
    m._elements[11] = _elements[0] * _elements[7] * _elements[9] + _elements[1] * _elements[4] * _elements[11] + _elements[3] * _elements[5] * _elements[8] - _elements[0] * _elements[5] * _elements[11] - _elements[1] * _elements[7] * _elements[8] - _elements[3] * _elements[4] * _elements[9];
    m._elements[12] = _elements[4] * _elements[10] * _elements[13] + _elements[5] * _elements[8] * _elements[14] + _elements[6] * _elements[9] * _elements[12] - _elements[4] * _elements[9] * _elements[14] - _elements[5] * _elements[10] * _elements[12] - _elements[6] * _elements[8] * _elements[13];
    m._elements[13] = _elements[0] * _elements[9] * _elements[14] + _elements[1] * _elements[10] * _elements[12] + _elements[2] * _elements[8] * _elements[13] - _elements[0] * _elements[10] * _elements[13] - _elements[1] * _elements[8] * _elements[14] - _elements[2] * _elements[9] * _elements[12];
    m._elements[14] = _elements[0] * _elements[6] * _elements[13] + _elements[1] * _elements[4] * _elements[14] + _elements[2] * _elements[5] * _elements[12] - _elements[0] * _elements[5] * _elements[14] - _elements[1] * _elements[6] * _elements[12] - _elements[2] * _elements[4] * _elements[13];
    m._elements[15] = _elements[0] * _elements[5] * _elements[10] + _elements[1] * _elements[6] * _elements[8] + _elements[2] * _elements[4] * _elements[9] - _elements[0] * _elements[6] * _elements[9] - _elements[1] * _elements[4] * _elements[10] - _elements[2] * _elements[5] * _elements[8];
    m.idiv(d);

    set(m);
}


// MARK: Complex getters
template <typename T>
T Matrix<T, 4, 4>::sum() const {
    T s = 0;
    for (int i = 0; i < 16; ++i) {
        s += _elements[i];
    }
    return s;
}

template <typename T>
T Matrix<T, 4, 4>::avg() const {
    return sum() / 16;
}

template <typename T>
T Matrix<T, 4, 4>::min() const {
    return minn(data(), 16);
}

template <typename T>
T Matrix<T, 4, 4>::max() const {
    return maxn(data(), 16);
}

template <typename T>
T Matrix<T, 4, 4>::absmin() const {
    return absminn(data(), 16);
}

template <typename T>
T Matrix<T, 4, 4>::absmax() const {
    return absmaxn(data(), 16);
}

template <typename T>
T Matrix<T, 4, 4>::trace() const {
    return _elements[0] + _elements[5] + _elements[10] + _elements[15];
}

template <typename T>
T Matrix<T, 4, 4>::determinant() const {
    return
          _elements[0] * _elements[5] * _elements[10] * _elements[15] + _elements[0] * _elements[6] * _elements[11] * _elements[13] + _elements[0] * _elements[7] * _elements[9] * _elements[14]
        + _elements[1] * _elements[4] * _elements[11] * _elements[14] + _elements[1] * _elements[6] * _elements[8] * _elements[15] + _elements[1] * _elements[7] * _elements[10] * _elements[12]
        + _elements[2] * _elements[4] * _elements[9] * _elements[15] + _elements[2] * _elements[5] * _elements[11] * _elements[12] + _elements[2] * _elements[7] * _elements[8] * _elements[13]
        + _elements[3] * _elements[4] * _elements[10] * _elements[13] + _elements[3] * _elements[5] * _elements[8] * _elements[14] + _elements[3] * _elements[6] * _elements[9] * _elements[12]
        - _elements[0] * _elements[5] * _elements[11] * _elements[14] - _elements[0] * _elements[6] * _elements[9] * _elements[15] - _elements[0] * _elements[7] * _elements[10] * _elements[13]
        - _elements[1] * _elements[4] * _elements[10] * _elements[15] - _elements[1] * _elements[6] * _elements[11] * _elements[12] - _elements[1] * _elements[7] * _elements[8] * _elements[14]
        - _elements[2] * _elements[4] * _elements[11] * _elements[13] - _elements[2] * _elements[5] * _elements[8] * _elements[15] - _elements[2] * _elements[7] * _elements[9] * _elements[12]
        - _elements[3] * _elements[4] * _elements[9] * _elements[14] - _elements[3] * _elements[5] * _elements[10] * _elements[12] - _elements[3] * _elements[6] * _elements[8] * _elements[13];
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::diagonal() const {
    return Matrix(
        _elements[0], 0, 0, 0,
        0, _elements[5], 0, 0,
        0, 0, _elements[10], 0,
        0, 0, 0, _elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::offDiagonal() const {
    return Matrix(
        0, _elements[1], _elements[2], _elements[3],
        _elements[4], 0, _elements[6], _elements[7],
        _elements[8], _elements[9], 0, _elements[11],
        _elements[12], _elements[13], _elements[14], 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::strictLowerTri() const {
    return Matrix(
        0, 0, 0, 0,
        _elements[4], 0, 0, 0,
        _elements[8], _elements[9], 0, 0,
        _elements[12], _elements[13], _elements[14], 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::strictUpperTri() const {
    return Matrix(
        0, _elements[1], _elements[2], _elements[3],
        0, 0, _elements[6], _elements[7],
        0, 0, 0, _elements[11],
        0, 0, 0, 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::lowerTri() const {
    return Matrix(
        _elements[0], 0, 0, 0,
        _elements[4], _elements[5], 0, 0,
        _elements[8], _elements[9], _elements[10], 0,
        _elements[12], _elements[13], _elements[14], _elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::upperTri() const {
    return Matrix(
        _elements[0], _elements[1], _elements[2], _elements[3],
        0, _elements[5], _elements[6], _elements[7],
        0, 0, _elements[10], _elements[11],
        0, 0, 0, _elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::transposed() const {
    return Matrix(
        _elements[0], _elements[4], _elements[8], _elements[12],
        _elements[1], _elements[5], _elements[9], _elements[13],
        _elements[2], _elements[6], _elements[10], _elements[14],
        _elements[3], _elements[7], _elements[11], _elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::inverse() const {
    Matrix m(*this);
    m.invert();
    return m;
}

template <typename T>
template <typename U>
Matrix<U, 4, 4> Matrix<T, 4, 4>::castTo() const {
    return Matrix<U, 4, 4>(
        static_cast<U>(_elements[0]),
        static_cast<U>(_elements[1]),
        static_cast<U>(_elements[2]),
        static_cast<U>(_elements[3]),
        static_cast<U>(_elements[4]),
        static_cast<U>(_elements[5]),
        static_cast<U>(_elements[6]),
        static_cast<U>(_elements[7]),
        static_cast<U>(_elements[8]),
        static_cast<U>(_elements[9]),
        static_cast<U>(_elements[10]),
        static_cast<U>(_elements[11]),
        static_cast<U>(_elements[12]),
        static_cast<U>(_elements[13]),
        static_cast<U>(_elements[14]),
        static_cast<U>(_elements[15]));
}


// MARK: Setter operators
template <typename T>
Matrix<T, 4, 4>& Matrix<T, 4, 4>::operator=(const Matrix& m) {
    set(m);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>& Matrix<T, 4, 4>::operator+=(T s) {
    iadd(s);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>& Matrix<T, 4, 4>::operator+=(const Matrix& m) {
    iadd(m);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>& Matrix<T, 4, 4>::operator-=(T s) {
    isub(s);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>& Matrix<T, 4, 4>::operator-=(const Matrix& m) {
    isub(m);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>& Matrix<T, 4, 4>::operator*=(T s) {
    imul(s);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>& Matrix<T, 4, 4>::operator*=(const Matrix& m) {
    imul(m);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>& Matrix<T, 4, 4>::operator/=(T s) {
    idiv(s);
    return *this;
}

template <typename T>
bool Matrix<T, 4, 4>::operator==(const Matrix& m) const {
    return _elements[0] == m._elements[0] && _elements[1] == m._elements[1] && _elements[2] == m._elements[2] &&
        _elements[3] == m._elements[3] && _elements[4] == m._elements[4] && _elements[5] == m._elements[5] &&
        _elements[6] == m._elements[6] && _elements[7] == m._elements[7] && _elements[8] == m._elements[8] &&
        _elements[9] == m._elements[9] && _elements[10] == m._elements[10] && _elements[11] == m._elements[11] &&
        _elements[12] == m._elements[12] && _elements[13] == m._elements[13] && _elements[14] == m._elements[14] &&
        _elements[15] == m._elements[15];
}

template <typename T>
bool Matrix<T, 4, 4>::operator!=(const Matrix& m) const {
    return _elements[0] != m._elements[0] || _elements[1] != m._elements[1] || _elements[2] != m._elements[2] ||
        _elements[3] != m._elements[3] || _elements[4] != m._elements[4] || _elements[5] != m._elements[5] ||
        _elements[6] != m._elements[6] || _elements[7] != m._elements[7] || _elements[8] != m._elements[8] ||
        _elements[9] != m._elements[9] || _elements[10] != m._elements[10] || _elements[11] != m._elements[11] ||
        _elements[12] != m._elements[12] || _elements[13] != m._elements[13] || _elements[14] != m._elements[14] ||
        _elements[15] != m._elements[15];
}


// MARK: Getter operators
template <typename T>
T& Matrix<T, 4, 4>::operator[](size_t i) {
    return _elements[i];
}

template <typename T>
const T& Matrix<T, 4, 4>::operator[](size_t i) const {
    return _elements[i];
}

template <typename T>
T& Matrix<T, 4, 4>::operator()(size_t i, size_t j) {
    return _elements[4 * i + j];
}

template <typename T>
const T& Matrix<T, 4, 4>::operator()(size_t i, size_t j) const {
    return _elements[4 * i + j];
}


// MARK: Helpers
template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::makeZero() {
    return Matrix(
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::makeIdentity() {
    return Matrix(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::makeScaleMatrix(T sx, T sy, T sz) {
    return Matrix(
        sx, 0, 0, 0,
        0, sy, 0, 0,
        0, 0, sz, 0,
        0, 0, 0, 1);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::makeScaleMatrix(const Vector<T, 3>& s) {
    return makeScaleMatrix(s.x, s.y, s.z);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::makeRotationMatrix(
    const Vector<T, 3>& axis, T rad) {
    return Matrix(Matrix<T, 3, 3>::makeRotationMatrix(axis, rad));
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::makeTranslationMatrix(const Vector<T, 3>& t) {
    return Matrix(
        1, 0, 0, t.x,
        0, 1, 0, t.y,
        0, 0, 1, t.z,
        0, 0, 0, 1);
}


// MARK: Operator overloadings
template <typename T>
Matrix<T, 4, 4> operator-(const Matrix<T, 4, 4>& a) {
    return a.mul(-1);
}

template <typename T>
Matrix<T, 4, 4> operator+(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b) {
    return a.add(b);
}

template <typename T>
Matrix<T, 4, 4> operator+(const Matrix<T, 4, 4>& a, T b) {
    return a.add(b);
}

template <typename T>
Matrix<T, 4, 4> operator+(T a, const Matrix<T, 4, 4>& b) {
    return b.radd(a);
}

template <typename T>
Matrix<T, 4, 4> operator-(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b) {
    return a.sub(b);
}

template <typename T>
Matrix<T, 4, 4> operator-(const Matrix<T, 4, 4>& a, T b) {
    return a.sub(b);
}

template <typename T>
Matrix<T, 4, 4> operator-(T a, const Matrix<T, 4, 4>& b) {
    return b.rsub(a);
}

template <typename T>
Matrix<T, 4, 4> operator*(const Matrix<T, 4, 4>& a, T b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 4, 4> operator*(T a, const Matrix<T, 4, 4>& b) {
    return b.rmul(a);
}

template <typename T>
Vector<T, 3> operator*(const Matrix<T, 4, 4>& a, const Vector<T, 3>& b) {
    return a.mul(b);
}

template <typename T>
Vector<T, 4> operator*(const Matrix<T, 4, 4>& a, const Vector<T, 4>& b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 4, 4> operator*(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b) {
    return a.mul(b);
}

template <typename T>
Matrix<T, 4, 4> operator/(const Matrix<T, 4, 4>& a, T b) {
    return a.div(b);
}

template <typename T>
Matrix<T, 4, 4> operator/(T a, const Matrix<T, 4, 4>& b) {
    return b.rdiv(a);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_MATRIX4X4_INL_H_
