// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_DETAIL_MATRIX4X4_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX4X4_INL_H_

#include <jet/math_utils.h>
#include <algorithm>
#include <cstring>

namespace jet {

// MARK: CTOR/DTOR
template <typename T>
Matrix<T, 4, 4>::Matrix() {
    elements[0] = 1; elements[4] = 0; elements[8] = 0; elements[12] = 0;
    elements[1] = 0; elements[5] = 1; elements[9] = 0; elements[13] = 0;
    elements[2] = 0; elements[6] = 0; elements[10] = 1; elements[14] = 0;
    elements[3] = 0; elements[7] = 0; elements[11] = 0; elements[15] = 1;
}

template <typename T>
Matrix<T, 4, 4>::Matrix(T s) {
    elements[0] = elements[4] = elements[8] = elements[12] =
    elements[1] = elements[5] = elements[9] = elements[13] =
    elements[2] = elements[6] = elements[10] = elements[14] =
    elements[3] = elements[7] = elements[11] = elements[15] = s;
}

template <typename T>
Matrix<T, 4, 4>::Matrix(
    T m00, T m10, T m20,
    T m01, T m11, T m21,
    T m02, T m12, T m22) {
    elements[0] = m00; elements[4] = m01; elements[8] = m02; elements[12] = 0;
    elements[1] = m10; elements[5] = m11; elements[9] = m12; elements[13] = 0;
    elements[2] = m20; elements[6] = m21; elements[10] = m22; elements[14] = 0;
    elements[3] = 0;   elements[7] = 0;   elements[11] = 0;   elements[15] = 1;
}

template <typename T>
Matrix<T, 4, 4>::Matrix(
    T m00, T m10, T m20, T m30,
    T m01, T m11, T m21, T m31,
    T m02, T m12, T m22, T m32,
    T m03, T m13, T m23, T m33) {
    elements[0] = m00;
    elements[4] = m01;
    elements[8] = m02;
    elements[12] = m03;
    elements[1] = m10;
    elements[5] = m11;
    elements[9] = m12;
    elements[13] = m13;
    elements[2] = m20;
    elements[6] = m21;
    elements[10] = m22;
    elements[14] = m23;
    elements[3] = m30;
    elements[7] = m31;
    elements[11] = m32;
    elements[15] = m33;
}

template <typename T>
Matrix<T, 4, 4>::Matrix(
    const Vector<T, 3>& col0,
    const Vector<T, 3>& col1,
    const Vector<T, 3>& col2) {
    elements[0] = col0.x;
    elements[4] = col0.y;
    elements[8] = col0.z;
    elements[12] = 0;
    elements[1] = col1.x;
    elements[5] = col1.y;
    elements[9] = col1.z;
    elements[13] = 0;
    elements[2] = col2.x;
    elements[6] = col2.y;
    elements[10] = col2.z;
    elements[14] = 0;
    elements[3] = 0;
    elements[7] = 0;
    elements[22] = 0;
    elements[15] = 1;
}

template <typename T>
Matrix<T, 4, 4>::Matrix(
    const Vector<T, 4>& col0,
    const Vector<T, 4>& col1,
    const Vector<T, 4>& col2,
    const Vector<T, 4>& col3) {
    elements[0] = col0.x;
    elements[1] = col0.y;
    elements[2] = col0.z;
    elements[3] = col0.w;
    elements[4] = col1.x;
    elements[5] = col1.y;
    elements[6] = col1.z;
    elements[7] = col1.w;
    elements[8] = col2.x;
    elements[9] = col2.y;
    elements[10] = col2.z;
    elements[11] = col2.w;
    elements[12] = col3.x;
    elements[13] = col3.y;
    elements[14] = col3.z;
    elements[15] = col3.w;
}

template <typename T>
Matrix<T, 4, 4>::Matrix(const Matrix<T, 3, 3>& m33) {
    set(m33);
}

template <typename T>
Matrix<T, 4, 4>::Matrix(const Matrix& m) {
    memcpy(elements, m.elements, sizeof(T) * 16);
}

template <typename T>
Matrix<T, 4, 4>::Matrix(const T* arr, size_t n) {
    memset(elements, 0, sizeof(T) * 16);
    memcpy(elements, arr, sizeof(T)*std::min(n, size_t(16)));
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
    T m00, T m10, T m20,
    T m01, T m11, T m21,
    T m02, T m12, T m22) {
    elements[0] = m00; elements[4] = m01; elements[8] = m02; elements[12] = 0;
    elements[1] = m10; elements[5] = m11; elements[9] = m12; elements[13] = 0;
    elements[2] = m20; elements[6] = m21; elements[10] = m22; elements[14] = 0;
    elements[3] = 0;   elements[7] = 0;   elements[11] = 0;   elements[15] = 1;
}

template <typename T>
void Matrix<T, 4, 4>::set(
    T m00, T m10, T m20, T m30,
    T m01, T m11, T m21, T m31,
    T m02, T m12, T m22, T m32,
    T m03, T m13, T m23, T m33) {
    elements[0] = m00;
    elements[4] = m01;
    elements[8] = m02;
    elements[12] = m03;
    elements[1] = m10;
    elements[5] = m11;
    elements[9] = m12;
    elements[13] = m13;
    elements[2] = m20;
    elements[6] = m21;
    elements[10] = m22;
    elements[14] = m23;
    elements[3] = m30;
    elements[7] = m31;
    elements[11] = m32;
    elements[15] = m33;
}

template <typename T>
void Matrix<T, 4, 4>::set(const Matrix<T, 3, 3>& m33) {
    set(m33.elements[0], m33.elements[1], m33.elements[2],
        m33.elements[3], m33.elements[4], m33.elements[5],
        m33.elements[6], m33.elements[7], m33.elements[8]);
}

template <typename T>
void Matrix<T, 4, 4>::set(const Matrix& m) {
    memcpy(elements, m.elements, sizeof(T) * 16);
}

template <typename T>
void Matrix<T, 4, 4>::set(const T* arr, size_t n) {
    memcpy(elements, arr, sizeof(T)*std::min(n, size_t(16)));
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
    elements[i] = row.x;
    elements[i + 4] = row.y;
    elements[i + 8] = row.z;
    elements[i + 12] = row.z;
}

template <typename T>
void Matrix<T, 4, 4>::setColumn(size_t j, const Vector<T, 4>& col) {
    elements[4 * j] = col.x;
    elements[4 * j + 1] = col.y;
    elements[4 * j + 2] = col.z;
    elements[4 * j + 3] = col.z;
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
    return elements;
}

template <typename T>
const T* Matrix<T, 4, 4>::data() const {
    return elements;
}

template <typename T>
Matrix<T, 3, 3> Matrix<T, 4, 4>::matrix3() const {
    return Matrix<T, 3, 3>(elements[0], elements[1], elements[2],
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
Vector<T, 3> Matrix<T, 4, 4>::mul(const Vector<T, 3>& v) const {
    // assuming affine transformation
    T invw = 1 / (elements[3] * v.x + elements[7] * v.y + elements[11] * v.z + elements[15]);
    return invw * Vector<T, 3>(
        elements[0] * v.x + elements[4] * v.y + elements[8] * v.z + elements[12],
        elements[1] * v.x + elements[5] * v.y + elements[9] * v.z + elements[13],
        elements[2] * v.x + elements[6] * v.y + elements[10] * v.z + elements[14]);
}

template <typename T>
Vector<T, 4> Matrix<T, 4, 4>::mul(const Vector<T, 4>& v) const {
    // assuming affine transformation
    T invw = 1 / (elements[3] * v.x + elements[7] * v.y + elements[11] * v.z + elements[15] * v.w);
    return invw*Vector<T, 4>(elements[0] * v.x + elements[4] * v.y + elements[8] * v.z + elements[12] * v.w,
        elements[1] * v.x + elements[5] * v.y + elements[9] * v.z + elements[13] * v.w,
        elements[2] * v.x + elements[6] * v.y + elements[10] * v.z + elements[14] * v.w,
        elements[3] * v.x + elements[7] * v.y + elements[11] * v.z + elements[15] * v.w);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::mul(const Matrix& m) const {
    return Matrix(elements[0] * m.elements[0] + elements[4] * m.elements[1] + elements[8] * m.elements[2] + elements[12] * m.elements[3],
        elements[1] * m.elements[0] + elements[5] * m.elements[1] + elements[9] * m.elements[2] + elements[13] * m.elements[3],
        elements[2] * m.elements[0] + elements[6] * m.elements[1] + elements[10] * m.elements[2] + elements[14] * m.elements[3],
        elements[3] * m.elements[0] + elements[7] * m.elements[1] + elements[11] * m.elements[2] + elements[15] * m.elements[3],

        elements[0] * m.elements[4] + elements[4] * m.elements[5] + elements[8] * m.elements[6] + elements[12] * m.elements[7],
        elements[1] * m.elements[4] + elements[5] * m.elements[5] + elements[9] * m.elements[6] + elements[13] * m.elements[7],
        elements[2] * m.elements[4] + elements[6] * m.elements[5] + elements[10] * m.elements[6] + elements[14] * m.elements[7],
        elements[3] * m.elements[4] + elements[7] * m.elements[5] + elements[11] * m.elements[6] + elements[15] * m.elements[7],

        elements[0] * m.elements[8] + elements[4] * m.elements[9] + elements[8] * m.elements[10] + elements[12] * m.elements[11],
        elements[1] * m.elements[8] + elements[5] * m.elements[9] + elements[9] * m.elements[10] + elements[13] * m.elements[11],
        elements[2] * m.elements[8] + elements[6] * m.elements[9] + elements[10] * m.elements[10] + elements[14] * m.elements[11],
        elements[3] * m.elements[8] + elements[7] * m.elements[9] + elements[11] * m.elements[10] + elements[15] * m.elements[11],

        elements[0] * m.elements[12] + elements[4] * m.elements[13] + elements[8] * m.elements[14] + elements[12] * m.elements[15],
        elements[1] * m.elements[12] + elements[5] * m.elements[13] + elements[9] * m.elements[14] + elements[13] * m.elements[15],
        elements[2] * m.elements[12] + elements[6] * m.elements[13] + elements[10] * m.elements[14] + elements[14] * m.elements[15],
        elements[3] * m.elements[12] + elements[7] * m.elements[13] + elements[11] * m.elements[14] + elements[15] * m.elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::div(T s) const {
    return Matrix(elements[0] / s, elements[1] / s, elements[2] / s, elements[3] / s,
        elements[4] / s, elements[5] / s, elements[6] / s, elements[7] / s,
        elements[8] / s, elements[9] / s, elements[10] / s, elements[11] / s,
        elements[12] / s, elements[13] / s, elements[14] / s, elements[15] / s);
}


// MARK: Binary operator methods - new instance = input (+) this instance
template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::radd(T s) const {
    return Matrix(s + elements[0], s + elements[1], s + elements[2], s + elements[3],
        s + elements[4], s + elements[5], s + elements[6], s + elements[7],
        s + elements[8], s + elements[9], s + elements[10], s + elements[11],
        s + elements[12], s + elements[13], s + elements[14], s + elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::radd(const Matrix& m) const {
    return Matrix(m.elements[0] + elements[0], m.elements[1] + elements[1], m.elements[2] + elements[2], m.elements[3] + elements[3],
        m.elements[4] + elements[4], m.elements[5] + elements[5], m.elements[6] + elements[6], m.elements[7] + elements[7],
        m.elements[8] + elements[8], m.elements[9] + elements[9], m.elements[10] + elements[10], m.elements[11] + elements[11],
        m.elements[12] + elements[12], m.elements[13] + elements[13], m.elements[14] + elements[14], m.elements[15] + elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rsub(T s) const {
    return Matrix(s - elements[0], s - elements[1], s - elements[2], s - elements[3],
        s - elements[4], s - elements[5], s - elements[6], s - elements[7],
        s - elements[8], s - elements[9], s - elements[10], s - elements[11],
        s - elements[12], s - elements[13], s - elements[14], s - elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rsub(const Matrix& m) const {
    return Matrix(m.elements[0] - elements[0], m.elements[1] - elements[1], m.elements[2] - elements[2], m.elements[3] - elements[3],
        m.elements[4] - elements[4], m.elements[5] - elements[5], m.elements[6] - elements[6], m.elements[7] - elements[7],
        m.elements[8] - elements[8], m.elements[9] - elements[9], m.elements[10] - elements[10], m.elements[11] - elements[11],
        m.elements[12] - elements[12], m.elements[13] - elements[13], m.elements[14] - elements[14], m.elements[15] - elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rmul(T s) const {
    return Matrix(s*elements[0], s*elements[1], s*elements[2], s*elements[3],
        s*elements[4], s*elements[5], s*elements[6], s*elements[7],
        s*elements[8], s*elements[9], s*elements[10], s*elements[11],
        s*elements[12], s*elements[13], s*elements[14], s*elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rmul(const Matrix& m) const {
    return Matrix(m.elements[0] * elements[0] + m.elements[4] * elements[1] + m.elements[8] * elements[2] + m.elements[12] * elements[3],
        m.elements[1] * elements[0] + m.elements[5] * elements[1] + m.elements[9] * elements[2] + m.elements[13] * elements[3],
        m.elements[2] * elements[0] + m.elements[6] * elements[1] + m.elements[10] * elements[2] + m.elements[14] * elements[3],
        m.elements[3] * elements[0] + m.elements[7] * elements[1] + m.elements[11] * elements[2] + m.elements[15] * elements[3],

        m.elements[0] * elements[4] + m.elements[4] * elements[5] + m.elements[8] * elements[6] + m.elements[12] * elements[7],
        m.elements[1] * elements[4] + m.elements[5] * elements[5] + m.elements[9] * elements[6] + m.elements[13] * elements[7],
        m.elements[2] * elements[4] + m.elements[6] * elements[5] + m.elements[10] * elements[6] + m.elements[14] * elements[7],
        m.elements[3] * elements[4] + m.elements[7] * elements[5] + m.elements[11] * elements[6] + m.elements[15] * elements[7],

        m.elements[0] * elements[8] + m.elements[4] * elements[9] + m.elements[8] * elements[10] + m.elements[12] * elements[11],
        m.elements[1] * elements[8] + m.elements[5] * elements[9] + m.elements[9] * elements[10] + m.elements[13] * elements[11],
        m.elements[2] * elements[8] + m.elements[6] * elements[9] + m.elements[10] * elements[10] + m.elements[14] * elements[11],
        m.elements[3] * elements[8] + m.elements[7] * elements[9] + m.elements[11] * elements[10] + m.elements[15] * elements[11],

        m.elements[0] * elements[12] + m.elements[4] * elements[13] + m.elements[8] * elements[14] + m.elements[12] * elements[15],
        m.elements[1] * elements[12] + m.elements[5] * elements[13] + m.elements[9] * elements[14] + m.elements[13] * elements[15],
        m.elements[2] * elements[12] + m.elements[6] * elements[13] + m.elements[10] * elements[14] + m.elements[14] * elements[15],
        m.elements[3] * elements[12] + m.elements[7] * elements[13] + m.elements[11] * elements[14] + m.elements[15] * elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::rdiv(T s) const {
    return Matrix(s / elements[0], s / elements[1], s / elements[2], s / elements[3],
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
    set(elements[0] * m.elements[0] + elements[4] * m.elements[1] + elements[8] * m.elements[2] + elements[12] * m.elements[3],
        elements[1] * m.elements[0] + elements[5] * m.elements[1] + elements[9] * m.elements[2] + elements[13] * m.elements[3],
        elements[2] * m.elements[0] + elements[6] * m.elements[1] + elements[10] * m.elements[2] + elements[14] * m.elements[3],
        elements[3] * m.elements[0] + elements[7] * m.elements[1] + elements[11] * m.elements[2] + elements[15] * m.elements[3],

        elements[0] * m.elements[4] + elements[4] * m.elements[5] + elements[8] * m.elements[6] + elements[12] * m.elements[7],
        elements[1] * m.elements[4] + elements[5] * m.elements[5] + elements[9] * m.elements[6] + elements[13] * m.elements[7],
        elements[2] * m.elements[4] + elements[6] * m.elements[5] + elements[10] * m.elements[6] + elements[14] * m.elements[7],
        elements[3] * m.elements[4] + elements[7] * m.elements[5] + elements[11] * m.elements[6] + elements[15] * m.elements[7],

        elements[0] * m.elements[8] + elements[4] * m.elements[9] + elements[8] * m.elements[10] + elements[12] * m.elements[11],
        elements[1] * m.elements[8] + elements[5] * m.elements[9] + elements[9] * m.elements[10] + elements[13] * m.elements[11],
        elements[2] * m.elements[8] + elements[6] * m.elements[9] + elements[10] * m.elements[10] + elements[14] * m.elements[11],
        elements[3] * m.elements[8] + elements[7] * m.elements[9] + elements[11] * m.elements[10] + elements[15] * m.elements[11],

        elements[0] * m.elements[12] + elements[4] * m.elements[13] + elements[8] * m.elements[14] + elements[12] * m.elements[15],
        elements[1] * m.elements[12] + elements[5] * m.elements[13] + elements[9] * m.elements[14] + elements[13] * m.elements[15],
        elements[2] * m.elements[12] + elements[6] * m.elements[13] + elements[10] * m.elements[14] + elements[14] * m.elements[15],
        elements[3] * m.elements[12] + elements[7] * m.elements[13] + elements[11] * m.elements[14] + elements[15] * m.elements[15]);
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
    m.elements[0] = elements[5] * elements[10] * elements[15] + elements[9] * elements[14] * elements[7] + elements[13] * elements[6] * elements[11] - elements[5] * elements[14] * elements[11] - elements[9] * elements[6] * elements[15] - elements[13] * elements[10] * elements[7];
    m.elements[1] = elements[1] * elements[14] * elements[11] + elements[9] * elements[2] * elements[15] + elements[13] * elements[10] * elements[3] - elements[1] * elements[10] * elements[15] - elements[9] * elements[14] * elements[3] - elements[13] * elements[2] * elements[11];
    m.elements[2] = elements[1] * elements[6] * elements[15] + elements[5] * elements[14] * elements[3] + elements[13] * elements[2] * elements[7] - elements[1] * elements[14] * elements[7] - elements[5] * elements[2] * elements[15] - elements[13] * elements[6] * elements[3];
    m.elements[3] = elements[1] * elements[10] * elements[7] + elements[5] * elements[2] * elements[11] + elements[9] * elements[6] * elements[3] - elements[1] * elements[6] * elements[11] - elements[5] * elements[10] * elements[3] - elements[9] * elements[2] * elements[7];
    m.elements[4] = elements[4] * elements[14] * elements[11] + elements[8] * elements[6] * elements[15] + elements[12] * elements[10] * elements[7] - elements[4] * elements[10] * elements[15] - elements[8] * elements[14] * elements[7] - elements[12] * elements[6] * elements[11];
    m.elements[5] = elements[0] * elements[10] * elements[15] + elements[8] * elements[14] * elements[3] + elements[12] * elements[2] * elements[11] - elements[0] * elements[14] * elements[11] - elements[8] * elements[2] * elements[15] - elements[12] * elements[10] * elements[3];
    m.elements[6] = elements[0] * elements[14] * elements[7] + elements[4] * elements[2] * elements[15] + elements[12] * elements[6] * elements[3] - elements[0] * elements[6] * elements[15] - elements[4] * elements[14] * elements[3] - elements[12] * elements[2] * elements[7];
    m.elements[7] = elements[0] * elements[6] * elements[11] + elements[4] * elements[10] * elements[3] + elements[8] * elements[2] * elements[7] - elements[0] * elements[10] * elements[7] - elements[4] * elements[2] * elements[11] - elements[8] * elements[6] * elements[3];
    m.elements[8] = elements[4] * elements[9] * elements[15] + elements[8] * elements[13] * elements[7] + elements[12] * elements[5] * elements[11] - elements[4] * elements[13] * elements[11] - elements[8] * elements[5] * elements[15] - elements[12] * elements[9] * elements[7];
    m.elements[9] = elements[0] * elements[13] * elements[11] + elements[8] * elements[1] * elements[15] + elements[12] * elements[9] * elements[3] - elements[0] * elements[9] * elements[15] - elements[8] * elements[13] * elements[3] - elements[12] * elements[1] * elements[11];
    m.elements[10] = elements[0] * elements[5] * elements[15] + elements[4] * elements[13] * elements[3] + elements[12] * elements[1] * elements[7] - elements[0] * elements[13] * elements[7] - elements[4] * elements[1] * elements[15] - elements[12] * elements[5] * elements[3];
    m.elements[11] = elements[0] * elements[9] * elements[7] + elements[4] * elements[1] * elements[11] + elements[8] * elements[5] * elements[3] - elements[0] * elements[5] * elements[11] - elements[4] * elements[9] * elements[3] - elements[8] * elements[1] * elements[7];
    m.elements[12] = elements[4] * elements[13] * elements[10] + elements[8] * elements[5] * elements[14] + elements[12] * elements[9] * elements[6] - elements[4] * elements[9] * elements[14] - elements[8] * elements[13] * elements[6] - elements[12] * elements[5] * elements[10];
    m.elements[13] = elements[0] * elements[9] * elements[14] + elements[8] * elements[13] * elements[2] + elements[12] * elements[1] * elements[10] - elements[0] * elements[13] * elements[10] - elements[8] * elements[1] * elements[14] - elements[12] * elements[9] * elements[2];
    m.elements[14] = elements[0] * elements[13] * elements[6] + elements[4] * elements[1] * elements[14] + elements[12] * elements[5] * elements[2] - elements[0] * elements[5] * elements[14] - elements[4] * elements[13] * elements[2] - elements[12] * elements[1] * elements[6];
    m.elements[15] = elements[0] * elements[5] * elements[10] + elements[4] * elements[9] * elements[2] + elements[8] * elements[1] * elements[6] - elements[0] * elements[9] * elements[6] - elements[4] * elements[1] * elements[10] - elements[8] * elements[5] * elements[2];
    m.idiv(d);

    set(m);
}


// MARK: Complex getters
template <typename T>
T Matrix<T, 4, 4>::sum() const {
    T s = 0;
    for (int i = 0; i < 16; ++i) s += elements[i];
    return s;
}

template <typename T>
T Matrix<T, 4, 4>::avg() const {
    return sum() / 16;
}

template <typename T>
T Matrix<T, 4, 4>::min() const {
    return min_n(elements, 16);
}

template <typename T>
T Matrix<T, 4, 4>::max() const {
    return max_n(elements, 16);
}

template <typename T>
T Matrix<T, 4, 4>::absmin() const {
    return absmin_n(elements, 16);
}

template <typename T>
T Matrix<T, 4, 4>::absmax() const {
    return absmax_n(elements, 16);
}

template <typename T>
T Matrix<T, 4, 4>::trace() const {
    return elements[0] + elements[5] + elements[10] + elements[15];
}

template <typename T>
T Matrix<T, 4, 4>::determinant() const {
    return elements[0] * elements[5] * elements[10] * elements[15] + elements[0] * elements[6] * elements[11] * elements[13] + elements[0] * elements[7] * elements[9] * elements[14]
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
    return Matrix(elements[0], 0, 0, 0,
        0, elements[5], 0, 0,
        0, 0, elements[10], 0,
        0, 0, 0, elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::offDiagonal() const {
    return Matrix(0, elements[1], elements[2], elements[3],
        elements[4], 0, elements[6], elements[7],
        elements[8], elements[9], 0, elements[11],
        elements[12], elements[13], elements[14], 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::strictLowerTri() const {
    return Matrix(0, elements[1], elements[2], elements[3],
        0, 0, elements[6], elements[7],
        0, 0, 0, elements[11],
        0, 0, 0, 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::strictUpperTri() const {
    return Matrix(0, 0, 0, 0,
        elements[4], 0, 0, 0,
        elements[8], elements[9], 0, 0,
        elements[12], elements[13], elements[14], 0);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::lowerTri() const {
    return Matrix(elements[0], elements[1], elements[2], elements[3],
        0, elements[5], elements[6], elements[7],
        0, 0, elements[10], elements[11],
        0, 0, 0, elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::upperTri() const {
    return Matrix(elements[0], 0, 0, 0,
        elements[4], elements[5], 0, 0,
        elements[8], elements[9], elements[10], 0,
        elements[12], elements[13], elements[14], elements[15]);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::transposed() const {
    return Matrix(elements[0], elements[4], elements[8], elements[12],
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
    return elements[4 * j + i];
}

template <typename T>
const T& Matrix<T, 4, 4>::operator()(size_t i, size_t j) const {
    return elements[4 * j + i];
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
Matrix<T, 4, 4> Matrix<T, 4, 4>::makeRotationMatrix(const Vector<T, 3>& axis, T rad) {
    return Matrix(
        1 + (1 - std::cos(rad))*(axis.x*axis.x - 1),
        axis.z*std::sin(rad) + (1 - std::cos(rad))*axis.x*axis.y,
        -axis.y*std::sin(rad) + (1 - std::cos(rad))*axis.x*axis.z,
        0,

        -axis.z*std::sin(rad) + (1 - std::cos(rad))*axis.x*axis.y,
        1 + (1 - std::cos(rad))*(axis.y*axis.y - 1),
        axis.x*std::sin(rad) + (1 - std::cos(rad))*axis.y*axis.z,
        0,

        axis.y*std::sin(rad) + (1 - std::cos(rad))*axis.x*axis.z,
        -axis.x*std::sin(rad) + (1 - std::cos(rad))*axis.y*axis.z,
        1 + (1 - std::cos(rad))*(axis.z*axis.z - 1),
        0,

        0,
        0,
        0,
        1);
}

template <typename T>
Matrix<T, 4, 4> Matrix<T, 4, 4>::makeTranslationMatrix(const Vector<T, 3>& t) {
    return Matrix(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        t.x, t.y, t.z, 1);
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
