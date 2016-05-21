// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_MATRIX3X3_H_
#define INCLUDE_JET_MATRIX3X3_H_

#include <jet/matrix.h>
#include <jet/vector3.h>
#include <limits>

namespace jet {

template <typename T>
class Matrix<T, 3, 3> {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "Matrix only can be instantiated with floating point types");

    T elements[9];

    // Constructors

    //! Default constructor.
    //! \warning This constructor will create identity matrix.
    Matrix();

    //! Constructs constant value matrix.
    explicit Matrix(T s);

    //! Constructs a matrix with input elements.
    //! \warning Ordering of the input elements is column-major.
    Matrix(
        T m00, T m10, T m20,
        T m01, T m11, T m21,
        T m02, T m12, T m22);

    //! Constructs a matrix with three column vectors.
    Matrix(
        const Vector3<T>& col0,
        const Vector3<T>& col1,
        const Vector3<T>& col2);

    //! Constructs a matrix with input matrix.
    Matrix(const Matrix& m);

    //! Constructs a matrix with input array.
    //! \warning Ordering of the input elements is column-major.
    Matrix(const T* arr, size_t n);


    // Basic setters

    //! Sets whole matrix with input scalar.
    void set(T s);

    //! Sets this matrix with input elements.
    //! \warning Ordering of the input elements is column-major.
    void set(
        T m00, T m10, T m20,
        T m01, T m11, T m21,
        T m02, T m12, T m22);

    //! Copies from input matrix.
    void set(const Matrix& m);

    //! Copies from input array.
    //! \warning Ordering of the input elements is column-major.
    void set(const T* arr, size_t n);

    //! Sets diagonal elements with input scalar.
    void setDiagonal(T s);

    //! Sets off-diagonal elements with input scalar.
    void setOffDiagonal(T s);

    //! Sets i-th row with input vector.
    void setRow(size_t i, const Vector3<T>& row);

    //! Sets i-th column with input vector.
    void setColumn(size_t i, const Vector3<T>& col);


    // Basic getters

    //! Returns true if this matrix is similar to the input matrix within the
    //! given tolerance.
    bool isSimilar(
        const Matrix& m,
        double tol = std::numeric_limits<double>::epsilon()) const;

    //! Returns true if this matrix is a square matrix.
    bool isSquare() const;

    //! Returns number of rows of this matrix.
    size_t rows() const;

    //! Returns number of columns of this matrix.
    size_t cols() const;

    //! Returns data pointer of this matrix.
    T* data();

    //! Returns constant pointer of this matrix.
    const T* data() const;


    // Binary operator methods - new instance = this instance (+) input
    //! Returns this matrix + input scalar.
    Matrix add(T s) const;

    //! Returns this matrix + input matrix (element-wise).
    Matrix add(const Matrix& m) const;

    //! Returns this matrix - input scalar.
    Matrix sub(T s) const;

    //! Returns this matrix - input matrix (element-wise).
    Matrix sub(const Matrix& m) const;

    //! Returns this matrix * input scalar.
    Matrix mul(T s) const;

    //! Returns this matrix * input vector.
    Vector3<T> mul(const Vector3<T>& v) const;

    //! Returns this matrix * input matrix.
    Matrix mul(const Matrix& m) const;

    //! Returns this matrix / input scalar.
    Matrix div(T s) const;


    // Binary operator methods - new instance = input (+) this instance
    //! Returns input scalar + this matrix.
    Matrix radd(T s) const;

    //! Returns input matrix + this matrix (element-wise).
    Matrix radd(const Matrix& m) const;

    //! Returns input scalar - this matrix.
    Matrix rsub(T s) const;

    //! Returns input matrix - this matrix (element-wise).
    Matrix rsub(const Matrix& m) const;

    //! Returns input scalar * this matrix.
    Matrix rmul(T s) const;

    //! Returns input matrix * this matrix.
    Matrix rmul(const Matrix& m) const;

    //! Returns input matrix / this scalar.
    Matrix rdiv(T s) const;


    // Augmented operator methods - this instance (+)= input
    //! Adds input scalar to this matrix.
    void iadd(T s);

    //! Adds input matrix to this matrix (element-wise).
    void iadd(const Matrix& m);

    //! Subtracts input scalar from this matrix.
    void isub(T s);

    //! Subtracts input matrix from this matrix (element-wise).
    void isub(const Matrix& m);

    //! Multiplies input scalar to this matrix.
    void imul(T s);

    //! Multiplies input matrix to this matrix.
    void imul(const Matrix& m);

    //! Divides this matrix with input scalar.
    void idiv(T s);


    // Modifiers
    //! Transposes this matrix.
    void transpose();

    //! Inverts this matrix.
    void invert();


    // Complex getters
    //! Returns sum of all elements.
    T sum() const;

    //! Returns average of all elements.
    T avg() const;

    //! Returns minimum among all elements.
    T min() const;

    //! Returns maximum among all elements.
    T max() const;

    //! Returns absolute minimum among all elements.
    T absmin() const;

    //! Returns absolute maximum among all elements.
    T absmax() const;

    //! Returns sum of all diagonal elements.
    T trace() const;

    //! Returns determinant of this matrix.
    T determinant() const;

    //! Returns diagonal part of this matrix.
    Matrix diagonal() const;

    //! Returns off-diagonal part of this matrix.
    Matrix offDiagonal() const;

    //! Returns strictly lower triangle part of this matrix.
    Matrix strictLowerTri() const;

    //! Returns strictly upper triangle part of this matrix.
    Matrix strictUpperTri() const;

    //! Returns lower triangle part of this matrix (including the diagonal).
    Matrix lowerTri() const;

    //! Returns upper triangle part of this matrix (including the diagonal).
    Matrix upperTri() const;

    //! Returns transposed matrix.
    Matrix transposed() const;

    //! Returns inverse matrix.
    Matrix inverse() const;

    template <typename U>
    Matrix<U, 3, 3> castTo() const;

    // Setter operators
    //! Assigns input matrix.
    Matrix& operator=(const Matrix& m);

    //! Addition assignment with input scalar.
    Matrix& operator+=(T s);

    //! Addition assignment with input matrix (element-wise).
    Matrix& operator+=(const Matrix& m);

    //! Subtraction assignment with input scalar.
    Matrix& operator-=(T s);

    //! Subtraction assignment with input matrix (element-wise).
    Matrix& operator-=(const Matrix& m);

    //! Multiplication assignment with input scalar.
    Matrix& operator*=(T s);

    //! Multiplication assignment with input matrix.
    Matrix& operator*=(const Matrix& m);

    //! Division assignment with input scalar.
    Matrix& operator/=(T s);


    // Getter operators
    //! Returns reference of i-th element.
    T& operator[](size_t i);

    //! Returns constant reference of i-th element.
    const T& operator[](size_t i) const;

    //! Returns reference of (i,j) element.
    T& operator()(size_t i, size_t j);

    //! Returns constant reference of (i,j) element.
    const T& operator()(size_t i, size_t j) const;

    //! Returns true if is equal to m.
    bool operator==(const Matrix& m) const;

    //! Returns true if is not equal to m.
    bool operator!=(const Matrix& m) const;


    // Helpers
    //! Sets all matrix entries to zero.
    static Matrix makeZero();

    //! Makes all diagonal elements to 1, and other elements to 0.
    static Matrix makeIdentity();

    //! Makes scale matrix.
    static Matrix makeScaleMatrix(T sx, T sy, T sz);

    //! Makes scale matrix.
    static Matrix makeScaleMatrix(const Vector3<T>& s);

    //! Makes rotation matrix.
    //! \warning Input angle should be radian.
    static Matrix makeRotationMatrix(const Vector3<T>& axis, T rad);
};

template <typename T> using Matrix3x3 = Matrix<T, 3, 3>;


// Operator overloadings
//! Returns a matrix with opposite sign.
template <typename T>
Matrix3x3<T> operator-(const Matrix3x3<T>& a);

//! Returns a + b (element-size).
template <typename T>
Matrix3x3<T> operator+(const Matrix3x3<T>& a, const Matrix3x3<T>& b);

//! Returns a + b', where every element of matrix b' is b.
template <typename T>
Matrix3x3<T> operator+(const Matrix3x3<T>& a, T b);

//! Returns a' + b, where every element of matrix a' is a.
template <typename T>
Matrix3x3<T> operator+(T a, const Matrix3x3<T>& b);

//! Returns a - b (element-size).
template <typename T>
Matrix3x3<T> operator-(const Matrix3x3<T>& a, const Matrix3x3<T>& b);

//! Returns a - b', where every element of matrix b' is b.
template <typename T>
Matrix3x3<T> operator-(const Matrix3x3<T>& a, T b);

//! Returns a' - b, where every element of matrix a' is a.
template <typename T>
Matrix3x3<T> operator-(T a, const Matrix3x3<T>& b);

//! Returns a * b', where every element of matrix b' is b.
template <typename T>
Matrix3x3<T> operator*(const Matrix3x3<T>& a, T b);

//! Returns a' * b, where every element of matrix a' is a.
template <typename T>
Matrix3x3<T> operator*(T a, const Matrix3x3<T>& b);

//! Returns a * b.
template <typename T>
Vector3<T> operator*(const Matrix3x3<T>& a, const Vector3<T>& b);

//! Returns a * b.
template <typename T>
Matrix3x3<T> operator*(const Matrix3x3<T>& a, const Matrix3x3<T>& b);

//! Returns a * b.
template <typename T>
Matrix3x3<T> operator*(const Matrix3x3<T>& a, const Matrix3x3<T>& b);

//! Returns a * b.
template <typename T>
Matrix3x3<T> operator*(const Matrix3x3<T>& a, const Matrix3x3<T>& b);

//! Returns a / b', where every element of matrix b' is b.
template <typename T>
Matrix3x3<T> operator/(const Matrix3x3<T>& a, T b);

//! Returns a' / b, where every element of matrix a' is a.
template <typename T>
Matrix3x3<T> operator/(T a, const Matrix3x3<T>& b);


typedef Matrix3x3<float> Matrix3x3F;
typedef Matrix3x3<double> Matrix3x3D;

}  // namespace jet

#include "detail/matrix3x3-inl.h"

#endif  // INCLUDE_JET_MATRIX3X3_H_
