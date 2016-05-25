// Copyright (c) 2016 Doyub Kim

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
    set(makeIdentity());
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
    elements[0] = elements[4] = elements[8] = elements[12] =
    elements[1] = elements[5] = elements[9] = elements[13] =
    elements[2] = elements[6] = elements[10] = elements[14] =
    elements[3] = elements[7] = elements[11] = elements[15] = s;
}

template <typename T>
void Matrix<T, 4, 4>::set(
    T m00, T m01, T m02,
    T m10, T m11, T m12,
    T m20, T m21, T m22) {
    elements[0] = m00; elements[1] = m01; elements[2] = m02;  elements[3] = 0;
    elements[4] = m10; elements[5] = m11; elements[6] = m12;  elements[7] = 0;
    elements[8] = m20; elements[9] = m21; elements[10] = m22; elements[11] = 0;
    elements[12] = 0;  elements[13] = 0;  elements[14] = 0;   elements[15] = 1;
}

template <typename T>
void Matrix<T, 4, 4>::set(
    T m00, T m01, T m02, T m03,
    T m10, T m11, T m12, T m13,
    T m20, T m21, T m22, T m23,
    T m30, T m31, T m32, T m33) {
    elements[0] = m00;
    elements[1] = m01;
    elements[2] = m02;
    elements[3] = m03;
    elements[4] = m10;
    elements[5] = m11;
    elements[6] = m12;
    elements[7] = m13;
    elements[8] = m20;
    elements[9] = m21;
    elements[10] = m22;
    elements[11] = m23;
    elements[12] = m30;
    elements[13] = m31;
    elements[14] = m32;
    elements[15] = m33;
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
    set(m33.elements[0], m33.elements[1], m33.elements[2], 0,
        m33.elements[3], m33.elements[4], m33.elements[5], 0,
        m33.elements[6], m33.elements[7], m33.elements[8], 0,
        0,               0,               0,               1);
}

template <typename T>
void Matrix<T, 4, 4>::set(const Matrix& m) {
    elements = m.elements;
}

template <typename T>
void Matrix<T, 4, 4>::set(const T* arr) {
    memcpy(elements.data(), arr, sizeof(T) * 16);
}

template <typename T>
void Matrix<T, 4, 4>::setDiagonal(T s) {
    elements[0] = elements[5] = elements[10] = elements[15] = s;
}

template <typename T>
void Matrix<T, 4, 4>::setOffDiagonal(T s) {
    elements[1] = elements[2] = elements[3] = elements[4]
    = elements[6] = elements[7] = elements[8] = elements[9]
    = elements[11] = elements[12] = elements[13] = elements[14] = s;
}

template <typename T>
void Matrix<T, 4, 4>::setRow(size_t i, const Vector<T, 4>& row) {
    elements[4 * i] = row.x;
    elements[4 * i + 1] = row.y;
    elements[4 * i + 2] = row.z;
    elements[4 * i + 3] = row.w;
}

template <typename T>
void Matrix<T, 4, 4>::setColumn(size_t j, const Vector<T, 4>& col) {
    elements[j] = col.x;
    elements[j + 4] = col.y;
    elements[j + 8] = col.z;
    elements[j + 12] = col.w;
}


// MARK: Basic getters
template <typename T>
bool Matrix<T, 4, 4>::isSimilar(const Matrix& m, double tol) const {
    return
        std::fabs(elements[0] - m.elements[0]) < tol
     && std::fabs(elements[1] - m.elements[1]) < tol
     && std::fabs(elements[2] - m.elements[2]) < tol
     && std::fabs(elements[3] - m.elements[3]) < tol
     && std::fabs(elements[4] - m.elements[4]) < tol
     && std::fabs(elements[5] - m.elements[5]) < tol
     && std::fabs(elements[6] - m.elements[6]) < tol
     && std::fabs(elements[7] - m.elements[7]) < tol
     && std::fabs(elements[8] - m.elements[8]) < tol
     && std::fabs(elements[9] - m.elements[9]) < tol
     && std::fabs(elements[10] - m.elements[10]) < tol
     && std::fabs(elements[11] - m.elements[11]) < tol
     && std::fabs(elements[12] - m.elements[12]) < tol
     && std::fabs(elements[13] - m.elements[13]) < tol
     && std::fabs(elements[14] - m.elements[14]) < tol
     && std::fabs(elements[15] - m.elements[15]) < tol;
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
    return elements.data();
}

template <typename T>
const T* Matrix<T, 4, 4>::data() const {
    return elements.data();
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 4, 4>::matrix3() const {
    return Matrix<T, 3, 3>(
        elements[0], elements[1], elements[2],
        elements[4], elements[5], elements[6],
        elements[8], elements[9], elements[10]);
}


// MARK: Binary operator methods - new instance = this instance (+) input
template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::add(T s) const {
    return Matrix(
        elements[0] + s,
        elements[1] + s,
        elements[2] + s,
        elements[3] + s,
        elements[4] + s,
        elements[5] + s,
        elements[6] + s,
        elements[7] + s,
        elements[8] + s,
        elements[9] + s,
        elements[10] + s,
        elements[11] + s,
        elements[12] + s,
        elements[13] + s,
        elements[14] + s,
        elements[15] + s);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::add(const Matrix& m) const {
    return Matrix(
        elements[0] + m.elements[0],
        elements[1] + m.elements[1],
        elements[2] + m.elements[2],
        elements[3] + m.elements[3],
        elements[4] + m.elements[4],
        elements[5] + m.elements[5],
        elements[6] + m.elements[6],
        elements[7] + m.elements[7],
        elements[8] + m.elements[8],
        elements[9] + m.elements[9],
        elements[10] + m.elements[10],
        elements[11] + m.elements[11],
        elements[12] + m.elements[12],
        elements[13] + m.elements[13],
        elements[14] + m.elements[14],
        elements[15] + m.elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::sub(T s) const {
    return Matrix(
        elements[0] - s, elements[1] - s, elements[2] - s, elements[3] - s,
        elements[4] - s, elements[5] - s, elements[6] - s, elements[7] - s,
        elements[8] - s, elements[9] - s, elements[10] - s, elements[11] - s,
        elements[12] - s, elements[13] - s, elements[14] - s, elements[15] - s);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::sub(const Matrix& m) const {
    return Matrix(
        elements[0] - m.elements[0],
        elements[1] - m.elements[1],
        elements[2] - m.elements[2],
        elements[3] - m.elements[3],
        elements[4] - m.elements[4],
        elements[5] - m.elements[5],
        elements[6] - m.elements[6],
        elements[7] - m.elements[7],
        elements[8] - m.elements[8],
        elements[9] - m.elements[9],
        elements[10] - m.elements[10],
        elements[11] - m.elements[11],
        elements[12] - m.elements[12],
        elements[13] - m.elements[13],
        elements[14] - m.elements[14],
        elements[15] - m.elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::mul(T s) const {
    return Matrix(
        elements[0] * s, elements[1] * s, elements[2] * s, elements[3] * s,
        elements[4] * s, elements[5] * s, elements[6] * s, elements[7] * s,
        elements[8] * s, elements[9] * s, elements[10] * s, elements[11] * s,
        elements[12] * s, elements[13] * s, elements[14] * s, elements[15] * s);
}

template <typename T>
Vector<T, 4> Matrix<T, 4, 4>::mul(const Vector<T, 4>& v) const {
    return Vector<T, 4>(
        elements[0] * v.x + elements[1] * v.y + elements[2] * v.z + elements[3] * v.w,
        elements[4] * v.x + elements[5] * v.y + elements[6] * v.z + elements[7] * v.w,
        elements[8] * v.x + elements[9] * v.y + elements[10] * v.z + elements[11] * v.w,
        elements[12] * v.x + elements[13] * v.y + elements[14] * v.z + elements[15] * v.w);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::mul(const Matrix& m) const {
    return Matrix(
        elements[0] * m.elements[0] + elements[1] * m.elements[4] + elements[2] * m.elements[8] + elements[3] * m.elements[12],
        elements[0] * m.elements[1] + elements[1] * m.elements[5] + elements[2] * m.elements[9] + elements[3] * m.elements[13],
        elements[0] * m.elements[2] + elements[1] * m.elements[6] + elements[2] * m.elements[10] + elements[3] * m.elements[14],
        elements[0] * m.elements[3] + elements[1] * m.elements[7] + elements[2] * m.elements[11] + elements[3] * m.elements[15],

        elements[4] * m.elements[0] + elements[5] * m.elements[4] + elements[6] * m.elements[8] + elements[7] * m.elements[12],
        elements[4] * m.elements[1] + elements[5] * m.elements[5] + elements[6] * m.elements[9] + elements[7] * m.elements[13],
        elements[4] * m.elements[2] + elements[5] * m.elements[6] + elements[6] * m.elements[10] + elements[7] * m.elements[14],
        elements[4] * m.elements[3] + elements[5] * m.elements[7] + elements[6] * m.elements[11] + elements[7] * m.elements[15],

        elements[8] * m.elements[0] + elements[9] * m.elements[4] + elements[10] * m.elements[8] + elements[11] * m.elements[12],
        elements[8] * m.elements[1] + elements[9] * m.elements[5] + elements[10] * m.elements[9] + elements[11] * m.elements[13],
        elements[8] * m.elements[2] + elements[9] * m.elements[6] + elements[10] * m.elements[10] + elements[11] * m.elements[14],
        elements[8] * m.elements[3] + elements[9] * m.elements[7] + elements[10] * m.elements[11] + elements[11] * m.elements[15],

        elements[12] * m.elements[0] + elements[13] * m.elements[4] + elements[14] * m.elements[8] + elements[15] * m.elements[12],
        elements[12] * m.elements[1] + elements[13] * m.elements[5] + elements[14] * m.elements[9] + elements[15] * m.elements[13],
        elements[12] * m.elements[2] + elements[13] * m.elements[6] + elements[14] * m.elements[10] + elements[15] * m.elements[14],
        elements[12] * m.elements[3] + elements[13] * m.elements[7] + elements[14] * m.elements[11] + elements[15] * m.elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::div(T s) const {
    return Matrix(
        elements[0] / s, elements[1] / s, elements[2] / s, elements[3] / s,
        elements[4] / s, elements[5] / s, elements[6] / s, elements[7] / s,
        elements[8] / s, elements[9] / s, elements[10] / s, elements[11] / s,
        elements[12] / s, elements[13] / s, elements[14] / s, elements[15] / s);
}


// MARK: Binary operator methods - new instance = input (+) this instance
template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::radd(T s) const {
    return Matrix(
        s + elements[0], s + elements[1], s + elements[2], s + elements[3],
        s + elements[4], s + elements[5], s + elements[6], s + elements[7],
        s + elements[8], s + elements[9], s + elements[10], s + elements[11],
        s + elements[12], s + elements[13], s + elements[14], s + elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::radd(const Matrix& m) const {
    return Matrix(
        m.elements[0] + elements[0], m.elements[1] + elements[1], m.elements[2] + elements[2], m.elements[3] + elements[3],
        m.elements[4] + elements[4], m.elements[5] + elements[5], m.elements[6] + elements[6], m.elements[7] + elements[7],
        m.elements[8] + elements[8], m.elements[9] + elements[9], m.elements[10] + elements[10], m.elements[11] + elements[11],
        m.elements[12] + elements[12], m.elements[13] + elements[13], m.elements[14] + elements[14], m.elements[15] + elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rsub(T s) const {
    return Matrix(
        s - elements[0], s - elements[1], s - elements[2], s - elements[3],
        s - elements[4], s - elements[5], s - elements[6], s - elements[7],
        s - elements[8], s - elements[9], s - elements[10], s - elements[11],
        s - elements[12], s - elements[13], s - elements[14], s - elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rsub(const Matrix& m) const {
    return Matrix(
        m.elements[0] - elements[0], m.elements[1] - elements[1], m.elements[2] - elements[2], m.elements[3] - elements[3],
        m.elements[4] - elements[4], m.elements[5] - elements[5], m.elements[6] - elements[6], m.elements[7] - elements[7],
        m.elements[8] - elements[8], m.elements[9] - elements[9], m.elements[10] - elements[10], m.elements[11] - elements[11],
        m.elements[12] - elements[12], m.elements[13] - elements[13], m.elements[14] - elements[14], m.elements[15] - elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rmul(T s) const {
    return Matrix(
        s*elements[0], s*elements[1], s*elements[2], s*elements[3],
        s*elements[4], s*elements[5], s*elements[6], s*elements[7],
        s*elements[8], s*elements[9], s*elements[10], s*elements[11],
        s*elements[12], s*elements[13], s*elements[14], s*elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rmul(const Matrix& m) const {
    return m.mul(*this);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rdiv(T s) const {
    return Matrix(
        s / elements[0], s / elements[1], s / elements[2], s / elements[3],
        s / elements[4], s / elements[5], s / elements[6], s / elements[7],
        s / elements[8], s / elements[9], s / elements[10], s / elements[11],
        s / elements[12], s / elements[13], s / elements[14], s / elements[15]);
}

// MARK: Augmented operator methods - this instance (+)= input
template <typename T>
void Matrix<T, 4, 4>::iadd(T s) {
    elements[0] += s; elements[1] += s; elements[2] += s; elements[3] += s;
    elements[4] += s; elements[5] += s; elements[6] += s; elements[7] += s;
    elements[8] += s; elements[9] += s; elements[10] += s; elements[11] += s;
    elements[12] += s; elements[13] += s; elements[14] += s; elements[15] += s;
}

template <typename T>
void Matrix<T, 4, 4>::iadd(const Matrix& m) {
    elements[0] += m.elements[0]; elements[1] += m.elements[1]; elements[2] += m.elements[2]; elements[3] += m.elements[3];
    elements[4] += m.elements[4]; elements[5] += m.elements[5]; elements[6] += m.elements[6]; elements[7] += m.elements[7];
    elements[8] += m.elements[8]; elements[9] += m.elements[9]; elements[10] += m.elements[10]; elements[11] += m.elements[11];
    elements[12] += m.elements[12]; elements[13] += m.elements[13]; elements[14] += m.elements[14]; elements[15] += m.elements[15];
}

template <typename T>
void Matrix<T, 4, 4>::isub(T s) {
    elements[0] -= s; elements[1] -= s; elements[2] -= s; elements[3] -= s;
    elements[4] -= s; elements[5] -= s; elements[6] -= s; elements[7] -= s;
    elements[8] -= s; elements[9] -= s; elements[10] -= s; elements[11] -= s;
    elements[12] -= s; elements[13] -= s; elements[14] -= s; elements[15] -= s;
}

template <typename T>
void Matrix<T, 4, 4>::isub(const Matrix& m) {
    elements[0] -= m.elements[0]; elements[1] -= m.elements[1]; elements[2] -= m.elements[2]; elements[3] -= m.elements[3];
    elements[4] -= m.elements[4]; elements[5] -= m.elements[5]; elements[6] -= m.elements[6]; elements[7] -= m.elements[7];
    elements[8] -= m.elements[8]; elements[9] -= m.elements[9]; elements[10] -= m.elements[10]; elements[11] -= m.elements[11];
    elements[12] -= m.elements[12]; elements[13] -= m.elements[13]; elements[14] -= m.elements[14]; elements[15] -= m.elements[15];
}

template <typename T>
void Matrix<T, 4, 4>::imul(T s) {
    elements[0] *= s; elements[1] *= s; elements[2] *= s; elements[3] *= s;
    elements[4] *= s; elements[5] *= s; elements[6] *= s; elements[7] *= s;
    elements[8] *= s; elements[9] *= s; elements[10] *= s; elements[11] *= s;
    elements[12] *= s; elements[13] *= s; elements[14] *= s; elements[15] *= s;
}

template <typename T>
void Matrix<T, 4, 4>::imul(const Matrix& m) {
    set(mul(m));
}

template <typename T>
void Matrix<T, 4, 4>::idiv(T s) {
    elements[0] /= s; elements[1] /= s; elements[2] /= s; elements[3] /= s;
    elements[4] /= s; elements[5] /= s; elements[6] /= s; elements[7] /= s;
    elements[8] /= s; elements[9] /= s; elements[10] /= s; elements[11] /= s;
    elements[12] /= s; elements[13] /= s; elements[14] /= s; elements[15] /= s;
}


// MARK: Modifiers
template <typename T>
void Matrix<T, 4, 4>::transpose() {
    std::swap(elements[1], elements[4]);
    std::swap(elements[2], elements[8]);
    std::swap(elements[3], elements[12]);
    std::swap(elements[6], elements[9]);
    std::swap(elements[7], elements[13]);
    std::swap(elements[11], elements[14]);
}

template <typename T>
void Matrix<T, 4, 4>::invert() {
    T d = determinant();
    Matrix m;
    m.elements[0] = elements[5] * elements[10] * elements[15] + elements[6] * elements[11] * elements[13] + elements[7] * elements[9] * elements[14] - elements[5] * elements[11] * elements[14] - elements[6] * elements[9] * elements[15] - elements[7] * elements[10] * elements[13];
    m.elements[1] = elements[1] * elements[11] * elements[14] + elements[2] * elements[9] * elements[15] + elements[3] * elements[10] * elements[13] - elements[1] * elements[10] * elements[15] - elements[2] * elements[11] * elements[13] - elements[3] * elements[9] * elements[14];
    m.elements[2] = elements[1] * elements[6] * elements[15] + elements[2] * elements[7] * elements[13] + elements[3] * elements[5] * elements[14] - elements[1] * elements[7] * elements[14] - elements[2] * elements[5] * elements[15] - elements[3] * elements[6] * elements[13];
    m.elements[3] = elements[1] * elements[7] * elements[10] + elements[2] * elements[5] * elements[11] + elements[3] * elements[6] * elements[9] - elements[1] * elements[6] * elements[11] - elements[2] * elements[7] * elements[9] - elements[3] * elements[5] * elements[10];
    m.elements[4] = elements[4] * elements[11] * elements[14] + elements[6] * elements[8] * elements[15] + elements[7] * elements[10] * elements[12] - elements[4] * elements[10] * elements[15] - elements[6] * elements[11] * elements[12] - elements[7] * elements[8] * elements[14];
    m.elements[5] = elements[0] * elements[10] * elements[15] + elements[2] * elements[11] * elements[12] + elements[3] * elements[8] * elements[14] - elements[0] * elements[11] * elements[14] - elements[2] * elements[8] * elements[15] - elements[3] * elements[10] * elements[12];
    m.elements[6] = elements[0] * elements[7] * elements[14] + elements[2] * elements[4] * elements[15] + elements[3] * elements[6] * elements[12] - elements[0] * elements[6] * elements[15] - elements[2] * elements[7] * elements[12] - elements[3] * elements[4] * elements[14];
    m.elements[7] = elements[0] * elements[6] * elements[11] + elements[2] * elements[7] * elements[8] + elements[3] * elements[4] * elements[10] - elements[0] * elements[7] * elements[10] - elements[2] * elements[4] * elements[11] - elements[3] * elements[6] * elements[8];
    m.elements[8] = elements[4] * elements[9] * elements[15] + elements[5] * elements[11] * elements[12] + elements[7] * elements[8] * elements[13] - elements[4] * elements[11] * elements[13] - elements[5] * elements[8] * elements[15] - elements[7] * elements[9] * elements[12];
    m.elements[9] = elements[0] * elements[11] * elements[13] + elements[1] * elements[8] * elements[15] + elements[3] * elements[9] * elements[12] - elements[0] * elements[9] * elements[15] - elements[1] * elements[11] * elements[12] - elements[3] * elements[8] * elements[13];
    m.elements[10] = elements[0] * elements[5] * elements[15] + elements[1] * elements[7] * elements[12] + elements[3] * elements[4] * elements[13] - elements[0] * elements[7] * elements[13] - elements[1] * elements[4] * elements[15] - elements[3] * elements[5] * elements[12];
    m.elements[11] = elements[0] * elements[7] * elements[9] + elements[1] * elements[4] * elements[11] + elements[3] * elements[5] * elements[8] - elements[0] * elements[5] * elements[11] - elements[1] * elements[7] * elements[8] - elements[3] * elements[4] * elements[9];
    m.elements[12] = elements[4] * elements[10] * elements[13] + elements[5] * elements[8] * elements[14] + elements[6] * elements[9] * elements[12] - elements[4] * elements[9] * elements[14] - elements[5] * elements[10] * elements[12] - elements[6] * elements[8] * elements[13];
    m.elements[13] = elements[0] * elements[9] * elements[14] + elements[1] * elements[10] * elements[12] + elements[2] * elements[8] * elements[13] - elements[0] * elements[10] * elements[13] - elements[1] * elements[8] * elements[14] - elements[2] * elements[9] * elements[12];
    m.elements[14] = elements[0] * elements[6] * elements[13] + elements[1] * elements[4] * elements[14] + elements[2] * elements[5] * elements[12] - elements[0] * elements[5] * elements[14] - elements[1] * elements[6] * elements[12] - elements[2] * elements[4] * elements[13];
    m.elements[15] = elements[0] * elements[5] * elements[10] + elements[1] * elements[6] * elements[8] + elements[2] * elements[4] * elements[9] - elements[0] * elements[6] * elements[9] - elements[1] * elements[4] * elements[10] - elements[2] * elements[5] * elements[8];
    m.idiv(d);

    set(m);
}


// MARK: Complex getters
template <typename T>
T Matrix<T, 4, 4>::sum() const {
    T s = 0;
    for (int i = 0; i < 16; ++i) {
        s += elements[i];
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
    return elements[0] + elements[5] + elements[10] + elements[15];
}

template <typename T>
T Matrix<T, 4, 4>::determinant() const {
    return
          elements[0] * elements[5] * elements[10] * elements[15] + elements[0] * elements[6] * elements[11] * elements[13] + elements[0] * elements[7] * elements[9] * elements[14]
        + elements[1] * elements[4] * elements[11] * elements[14] + elements[1] * elements[6] * elements[8] * elements[15] + elements[1] * elements[7] * elements[10] * elements[12]
        + elements[2] * elements[4] * elements[9] * elements[15] + elements[2] * elements[5] * elements[11] * elements[12] + elements[2] * elements[7] * elements[8] * elements[13]
        + elements[3] * elements[4] * elements[10] * elements[13] + elements[3] * elements[5] * elements[8] * elements[14] + elements[3] * elements[6] * elements[9] * elements[12]
        - elements[0] * elements[5] * elements[11] * elements[14] - elements[0] * elements[6] * elements[9] * elements[15] - elements[0] * elements[7] * elements[10] * elements[13]
        - elements[1] * elements[4] * elements[10] * elements[15] - elements[1] * elements[6] * elements[11] * elements[12] - elements[1] * elements[7] * elements[8] * elements[14]
        - elements[2] * elements[4] * elements[11] * elements[13] - elements[2] * elements[5] * elements[8] * elements[15] - elements[2] * elements[7] * elements[9] * elements[12]
        - elements[3] * elements[4] * elements[9] * elements[14] - elements[3] * elements[5] * elements[10] * elements[12] - elements[3] * elements[6] * elements[8] * elements[13];
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::diagonal() const {
    return Matrix(
        elements[0], 0, 0, 0,
        0, elements[5], 0, 0,
        0, 0, elements[10], 0,
        0, 0, 0, elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::offDiagonal() const {
    return Matrix(
        0, elements[1], elements[2], elements[3],
        elements[4], 0, elements[6], elements[7],
        elements[8], elements[9], 0, elements[11],
        elements[12], elements[13], elements[14], 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::strictLowerTri() const {
    return Matrix(
        0, 0, 0, 0,
        elements[4], 0, 0, 0,
        elements[8], elements[9], 0, 0,
        elements[12], elements[13], elements[14], 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::strictUpperTri() const {
    return Matrix(
        0, elements[1], elements[2], elements[3],
        0, 0, elements[6], elements[7],
        0, 0, 0, elements[11],
        0, 0, 0, 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::lowerTri() const {
    return Matrix(
        elements[0], 0, 0, 0,
        elements[4], elements[5], 0, 0,
        elements[8], elements[9], elements[10], 0,
        elements[12], elements[13], elements[14], elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::upperTri() const {
    return Matrix(
        elements[0], elements[1], elements[2], elements[3],
        0, elements[5], elements[6], elements[7],
        0, 0, elements[10], elements[11],
        0, 0, 0, elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::transposed() const {
    return Matrix(
        elements[0], elements[4], elements[8], elements[12],
        elements[1], elements[5], elements[9], elements[13],
        elements[2], elements[6], elements[10], elements[14],
        elements[3], elements[7], elements[11], elements[15]);
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
        static_cast<U>(elements[0]),
        static_cast<U>(elements[1]),
        static_cast<U>(elements[2]),
        static_cast<U>(elements[3]),
        static_cast<U>(elements[4]),
        static_cast<U>(elements[5]),
        static_cast<U>(elements[6]),
        static_cast<U>(elements[7]),
        static_cast<U>(elements[8]),
        static_cast<U>(elements[9]),
        static_cast<U>(elements[10]),
        static_cast<U>(elements[11]),
        static_cast<U>(elements[12]),
        static_cast<U>(elements[13]),
        static_cast<U>(elements[14]),
        static_cast<U>(elements[15]));
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
    return elements[0] == m.elements[0] && elements[1] == m.elements[1] && elements[2] == m.elements[2] &&
        elements[3] == m.elements[3] && elements[4] == m.elements[4] && elements[5] == m.elements[5] &&
        elements[6] == m.elements[6] && elements[7] == m.elements[7] && elements[8] == m.elements[8] &&
        elements[9] == m.elements[9] && elements[10] == m.elements[10] && elements[11] == m.elements[11] &&
        elements[12] == m.elements[12] && elements[13] == m.elements[13] && elements[14] == m.elements[14] &&
        elements[15] == m.elements[15];
}

template <typename T>
bool Matrix<T, 4, 4>::operator!=(const Matrix& m) const {
    return elements[0] != m.elements[0] || elements[1] != m.elements[1] || elements[2] != m.elements[2] ||
        elements[3] != m.elements[3] || elements[4] != m.elements[4] || elements[5] != m.elements[5] ||
        elements[6] != m.elements[6] || elements[7] != m.elements[7] || elements[8] != m.elements[8] ||
        elements[9] != m.elements[9] || elements[10] != m.elements[10] || elements[11] != m.elements[11] ||
        elements[12] != m.elements[12] || elements[13] != m.elements[13] || elements[14] != m.elements[14] ||
        elements[15] != m.elements[15];
}


// MARK: Getter operators
template <typename T>
T& Matrix<T, 4, 4>::operator[](size_t i) {
    return elements[i];
}

template <typename T>
const T& Matrix<T, 4, 4>::operator[](size_t i) const {
    return elements[i];
}

template <typename T>
T& Matrix<T, 4, 4>::operator()(size_t i, size_t j) {
    return elements[4 * i + j];
}

template <typename T>
const T& Matrix<T, 4, 4>::operator()(size_t i, size_t j) const {
    return elements[4 * i + j];
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
