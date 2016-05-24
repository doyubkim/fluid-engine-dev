// Copyright (c) 2016 Doyub Kim

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
    set(makeIdentity());
}

template <typename T>
Matrix<T, 2, 2>::Matrix(T s) {
    set(s);
}

template <typename T>
Matrix<T, 2, 2>::Matrix(
    T m00, T m01,
    T m10, T m11) {
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
    elements[0] = elements[1] = elements[2] = elements[3] = s;
}

template <typename T>
void Matrix<T, 2, 2>::set(
    T m00, T m01,
    T m10, T m11) {
    elements[0] = m00;
    elements[1] = m01;
    elements[2] = m10;
    elements[3] = m11;
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
    memcpy(elements.data(), m.elements.data(), sizeof(T) * 4);
}

template <typename T>
void Matrix<T, 2, 2>::set(const T* arr) {
    memcpy(elements.data(), arr, sizeof(T) * 4);
}

template <typename T>
void Matrix<T, 2, 2>::setDiagonal(T s) {
    elements[0] = elements[3] = s;
}

template <typename T>
void Matrix<T, 2, 2>::setOffDiagonal(T s) {
    elements[1] = elements[2] = s;
}

template <typename T>
void Matrix<T, 2, 2>::setRow(size_t i, const Vector<T, 2>& row) {
    elements[2 * i] = row.x;
    elements[2 * i + 1] = row.y;
}

template <typename T>
void Matrix<T, 2, 2>::setColumn(size_t j, const Vector<T, 2>& col) {
    elements[j] = col.x;
    elements[j + 2] = col.y;
}


// MARK: Basic getters
template <typename T>
bool Matrix<T, 2, 2>::isSimilar(const Matrix& m, double tol) const {
    return (std::fabs(elements[0] - m.elements[0]) < tol)
        && (std::fabs(elements[1] - m.elements[1]) < tol)
        && (std::fabs(elements[2] - m.elements[2]) < tol)
        && (std::fabs(elements[3] - m.elements[3]) < tol);
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
    return elements.data();
}

template <typename T>
const T* Matrix<T, 2, 2>::data() const {
    return elements.data();
}


// MARK: Binary operator methods - new instance = this instance (+) input
template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::add(T s) const {
    return Matrix(
        elements[0] + s,
        elements[1] + s,
        elements[2] + s,
        elements[3] + s);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::add(const Matrix& m) const {
    return Matrix(
        elements[0] + m.elements[0],
        elements[1] + m.elements[1],
        elements[2] + m.elements[2],
        elements[3] + m.elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::sub(T s) const {
    return Matrix(
        elements[0] - s,
        elements[1] - s,
        elements[2] - s,
        elements[3] - s);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::sub(const Matrix& m) const {
    return Matrix(
        elements[0] - m.elements[0],
        elements[1] - m.elements[1],
        elements[2] - m.elements[2],
        elements[3] - m.elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::mul(T s) const {
    return Matrix(
        elements[0] * s,
        elements[1] * s,
        elements[2] * s,
        elements[3] * s);
}

template <typename T>
Vector<T, 2> Matrix<T, 2, 2>::mul(const Vector<T, 2>& v) const {
    return Vector<T, 2>(elements[0] * v.x + elements[1] * v.y,
                        elements[2] * v.x + elements[3] * v.y);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::mul(const Matrix& m) const {
    return Matrix(
        elements[0] * m.elements[0] + elements[1] * m.elements[2],
        elements[0] * m.elements[1] + elements[1] * m.elements[3],
        elements[2] * m.elements[0] + elements[3] * m.elements[2],
        elements[2] * m.elements[1] + elements[3] * m.elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::div(T s) const {
    return Matrix(
        elements[0] / s,
        elements[1] / s,
        elements[2] / s,
        elements[3] / s);
}


// MARK: Binary operator methods - new instance = input (+) this instance
template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::radd(T s) const {
    return Matrix(
        s + elements[0],
        s + elements[1],
        s + elements[2],
        s + elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::radd(const Matrix& m) const {
    return Matrix(
        m.elements[0] + elements[0],
        m.elements[1] + elements[1],
        m.elements[2] + elements[2],
        m.elements[3] + elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rsub(T s) const {
    return Matrix(
        s - elements[0],
        s - elements[1],
        s - elements[2],
        s - elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rsub(const Matrix& m) const {
    return Matrix(
        m.elements[0] - elements[0],
        m.elements[1] - elements[1],
        m.elements[2] - elements[2],
        m.elements[3] - elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rmul(T s) const {
    return Matrix(s*elements[0], s*elements[1], s*elements[2], s*elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rmul(const Matrix& m) const {
    return m.mul(*this);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::rdiv(T s) const {
    return Matrix(
        s / elements[0],
        s / elements[1],
        s / elements[2],
        s / elements[3]);
}

// MARK: Augmented operator methods - this instance (+)= input
template <typename T>
void Matrix<T, 2, 2>::iadd(T s) {
    elements[0] += s;
    elements[1] += s;
    elements[2] += s;
    elements[3] += s;
}

template <typename T>
void Matrix<T, 2, 2>::iadd(const Matrix& m) {
    elements[0] += m.elements[0];
    elements[1] += m.elements[1];
    elements[2] += m.elements[2];
    elements[3] += m.elements[3];
}

template <typename T>
void Matrix<T, 2, 2>::isub(T s) {
    elements[0] -= s;
    elements[1] -= s;
    elements[2] -= s;
    elements[3] -= s;
}

template <typename T>
void Matrix<T, 2, 2>::isub(const Matrix& m) {
    elements[0] -= m.elements[0];
    elements[1] -= m.elements[1];
    elements[2] -= m.elements[2];
    elements[3] -= m.elements[3];
}

template <typename T>
void Matrix<T, 2, 2>::imul(T s) {
    elements[0] *= s;
    elements[1] *= s;
    elements[2] *= s;
    elements[3] *= s;
}

template <typename T>
void Matrix<T, 2, 2>::imul(const Matrix& m) {
    set(mul(m));
}

template <typename T>
void Matrix<T, 2, 2>::idiv(T s) {
    elements[0] /= s;
    elements[1] /= s;
    elements[2] /= s;
    elements[3] /= s;
}


// MARK: Modifiers
template <typename T>
void Matrix<T, 2, 2>::transpose() {
    std::swap(elements[1], elements[2]);
}

template <typename T>
void Matrix<T, 2, 2>::invert() {
    T d = determinant();
    Matrix m;
    m.elements[0] = elements[3];
    m.elements[1] = -elements[1];
    m.elements[2] = -elements[2];
    m.elements[3] = elements[0];
    m.idiv(d);

    set(m);
}


// MARK: Complex getters
template <typename T>
T Matrix<T, 2, 2>::sum() const {
    T s = 0;
    for (int i = 0; i < 4; ++i) {
        s += elements[i];
    }
    return s;
}

template <typename T>
T Matrix<T, 2, 2>::avg() const {
    return sum() / 4;
}

template <typename T>
T Matrix<T, 2, 2>::min() const {
    return std::min(
        std::min(elements[0], elements[1]), std::min(elements[2], elements[3]));
}

template <typename T>
T Matrix<T, 2, 2>::max() const {
    return std::max(
        std::max(elements[0], elements[1]), std::max(elements[2], elements[3]));
}

template <typename T>
T Matrix<T, 2, 2>::absmin() const {
    return ::jet::absmin(
        ::jet::absmin(elements[0], elements[1]),
        ::jet::absmin(elements[2], elements[3]));
}

template <typename T>
T Matrix<T, 2, 2>::absmax() const {
    return ::jet::absmax(
        ::jet::absmax(elements[0], elements[1]),
        ::jet::absmax(elements[2], elements[3]));
}

template <typename T>
T Matrix<T, 2, 2>::trace() const {
    return elements[0] + elements[3];
}

template <typename T>
T Matrix<T, 2, 2>::determinant() const {
    return elements[0] * elements[3] - elements[1] * elements[2];
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::diagonal() const {
    return Matrix(
        elements[0], 0,
        0, elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::offDiagonal() const {
    return Matrix(
        0, elements[1],
        elements[2], 0);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::strictLowerTri() const {
    return Matrix(
        0, 0,
        elements[2], 0);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::strictUpperTri() const {
    return Matrix(
        0, elements[1],
        0, 0);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::lowerTri() const {
    return Matrix(
        elements[0], 0,
        elements[2], elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::upperTri() const {
    return Matrix(
        elements[0], elements[1],
        0, elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::transposed() const {
    return Matrix(
        elements[0], elements[2],
        elements[1], elements[3]);
}

template <typename T>
Matrix<T, 2, 2> Matrix<T, 2, 2>::inverse() const {
    Matrix m(*this);
    m.invert();
    return m;
}

template <typename T>
template <typename U>
Matrix<U, 2, 2> Matrix<T, 2, 2>::castTo() const {
    return Matrix<U, 2, 2>(
        static_cast<U>(elements[0]),
        static_cast<U>(elements[1]),
        static_cast<U>(elements[2]),
        static_cast<U>(elements[3]));
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
    return elements[0] == m.elements[0] && elements[1] == m.elements[1] &&
           elements[2] == m.elements[2] && elements[3] == m.elements[3];
}

template <typename T>
bool Matrix<T, 2, 2>::operator!=(const Matrix& m) const {
    return elements[0] != m.elements[0] || elements[1] != m.elements[1] ||
           elements[2] != m.elements[2] || elements[3] != m.elements[3];
}


// MARK: Getter operators
template <typename T>
T& Matrix<T, 2, 2>::operator[](size_t i) {
    return elements[i];
}

template <typename T>
const T& Matrix<T, 2, 2>::operator[](size_t i) const {
    return elements[i];
}

template <typename T>
T& Matrix<T, 2, 2>::operator()(size_t i, size_t j) {
    return elements[2 * i + j];
}

template <typename T>
const T& Matrix<T, 2, 2>::operator()(size_t i, size_t j) const {
    return elements[2 * i + j];
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
