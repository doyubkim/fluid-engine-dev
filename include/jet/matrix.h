// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_H_
#define INCLUDE_JET_MATRIX_H_

#include <jet/macros.h>
#include <jet/matrix_expression.h>

#include <array>
#include <type_traits>

namespace jet {

//!
//! \brief Static-sized M x N matrix class.
//!
//! This class defines M x N row-major matrix data where its size is determined
//! statically at compile time.
//!
//! \tparam T - Real number type.
//! \tparam M - Number of rows.
//! \tparam N - Number of columns.
//!
template <typename T, size_t M, size_t N>
class Matrix final : public MatrixExpression<T, Matrix<T, M, N>> {
 public:
    static_assert(
        M > 0,
        "Number of rows for static-sized matrix should be greater than zero.");
    static_assert(
        N > 0,
        "Number of columns for static-sized matrix should be greater than "
        "zero.");
    static_assert(!(M == 2 && N == 2) && !(M == 3 && N == 3) &&
                      !(M == 4 && N == 4),
                  "Use specialized matrix for 2z2, 3x3, and 4x4 matricies.");
    static_assert(std::is_floating_point<T>::value,
                  "Matrix only can be instantiated with floating point types");

    typedef std::array<T, M * N> ContainerType;
    typedef typename ContainerType::iterator Iterator;
    typedef typename ContainerType::const_iterator ConstIterator;

    //! Default constructor.
    //! \warning This constructor will create zero matrix.
    Matrix();

    //! Constructs matrix instance with parameters.
    template <typename... Params>
    explicit Matrix(Params... params);

    //!
    //! \brief Constructs a matrix with given initializer list \p lst.
    //!
    //! This constructor will build a matrix with given initializer list \p lst
    //! such as
    //!
    //! \code{.cpp}
    //! Matrix<float, 3, 4> mat = {
    //!     {1.f, 2.f, 4.f, 3.f},
    //!     {9.f, 3.f, 5.f, 1.f},
    //!     {4.f, 8.f, 1.f, 5.f}
    //! };
    //! \endcode
    //!
    //! \param lst Initializer list that should be copy to the new matrix.
    //!
    Matrix(const std::initializer_list<std::initializer_list<T>>& lst);

    //! Constructs a matrix with expression template.
    template <typename E>
    Matrix(const MatrixExpression<T, E>& other);

    //! Copy constructor.
    Matrix(const Matrix& other);

    // MARK: Basic setters

    //! Resizes to m x n matrix with initial value \p s.
    void resize(size_t m, size_t n, const T& s = T(0));

    //! Sets whole matrix with input scalar.
    void set(const T& s);

    //!
    //! \brief Sets a matrix with given initializer list \p lst.
    //!
    //! This function will fill the matrix with given initializer list \p lst
    //! such as
    //!
    //! \code{.cpp}
    //! Matrix<float, 3, 4> mat;
    //! mat.set({
    //!     {1.f, 2.f, 4.f, 3.f},
    //!     {9.f, 3.f, 5.f, 1.f},
    //!     {4.f, 8.f, 1.f, 5.f}
    //! });
    //! \endcode
    //!
    //! \param lst Initializer list that should be copy to the new matrix.
    //!
    void set(const std::initializer_list<std::initializer_list<T>>& lst);

    //! Copies from input matrix expression.
    template <typename E>
    void set(const MatrixExpression<T, E>& other);

    //! Sets diagonal elements with input scalar.
    void setDiagonal(const T& s);

    //! Sets off-diagonal elements with input scalar.
    void setOffDiagonal(const T& s);

    //! Sets i-th row with input vector.
    template <typename E>
    void setRow(size_t i, const VectorExpression<T, E>& row);

    //! Sets j-th column with input vector.
    template <typename E>
    void setColumn(size_t j, const VectorExpression<T, E>& col);

    // MARK: Basic getters
    template <typename E>
    bool isEqual(const MatrixExpression<T, E>& other) const;

    //! Returns true if this matrix is similar to the input matrix within the
    //! given tolerance.
    template <typename E>
    bool isSimilar(const MatrixExpression<T, E>& other,
                   double tol = std::numeric_limits<double>::epsilon()) const;

    //! Returns true if this matrix is a square matrix.
    constexpr bool isSquare() const;

    //! Returns the size of this matrix.
    constexpr Size2 size() const;

    //! Returns number of rows of this matrix.
    constexpr size_t rows() const;

    //! Returns number of columns of this matrix.
    constexpr size_t cols() const;

    //! Returns data pointer of this matrix.
    T* data();

    //! Returns constant pointer of this matrix.
    const T* const data() const;

    //! Returns the begin iterator of the matrix.
    Iterator begin();

    //! Returns the begin const iterator of the matrix.
    ConstIterator begin() const;

    //! Returns the end iterator of the matrix.
    Iterator end();

    //! Returns the end const iterator of the matrix.
    ConstIterator end() const;

    // MARK: Binary operator methods - new instance = this instance (+) input

    //! Returns this matrix + input scalar.
    MatrixScalarAdd<T, Matrix> add(const T& s) const;

    //! Returns this matrix + input matrix (element-wise).
    template <typename E>
    MatrixAdd<T, Matrix, E> add(const E& m) const;

    //! Returns this matrix - input scalar.
    MatrixScalarSub<T, Matrix> sub(const T& s) const;

    //! Returns this matrix - input matrix (element-wise).
    template <typename E>
    MatrixSub<T, Matrix, E> sub(const E& m) const;

    //! Returns this matrix * input scalar.
    MatrixScalarMul<T, Matrix> mul(const T& s) const;

    //! Returns this matrix * input vector.
    template <typename VE>
    MatrixVectorMul<T, Matrix, VE> mul(const VectorExpression<T, VE>& v) const;

    //! Returns this matrix * input matrix.
    template <size_t L>
    MatrixMul<T, Matrix, Matrix<T, N, L>> mul(const Matrix<T, N, L>& m) const;

    //! Returns this matrix / input scalar.
    MatrixScalarDiv<T, Matrix> div(const T& s) const;

    // MARK: Binary operator methods - new instance = input (+) this instance
    //! Returns input scalar + this matrix.
    MatrixScalarAdd<T, Matrix> radd(const T& s) const;

    //! Returns input matrix + this matrix (element-wise).
    template <typename E>
    MatrixAdd<T, Matrix, E> radd(const E& m) const;

    //! Returns input scalar - this matrix.
    MatrixScalarRSub<T, Matrix> rsub(const T& s) const;

    //! Returns input matrix - this matrix (element-wise).
    template <typename E>
    MatrixSub<T, Matrix, E> rsub(const E& m) const;

    //! Returns input scalar * this matrix.
    MatrixScalarMul<T, Matrix> rmul(const T& s) const;

    //! Returns input matrix * this matrix.
    template <size_t L>
    MatrixMul<T, Matrix<T, N, L>, Matrix> rmul(const Matrix<T, N, L>& m) const;

    //! Returns input matrix / this scalar.
    MatrixScalarRDiv<T, Matrix> rdiv(const T& s) const;

    // MARK: Augmented operator methods - this instance (+)= input

    //! Adds input scalar to this matrix.
    void iadd(const T& s);

    //! Adds input matrix to this matrix (element-wise).
    template <typename E>
    void iadd(const E& m);

    //! Subtracts input scalar from this matrix.
    void isub(const T& s);

    //! Subtracts input matrix from this matrix (element-wise).
    template <typename E>
    void isub(const E& m);

    //! Multiplies input scalar to this matrix.
    void imul(const T& s);

    //! Multiplies input matrix to this matrix.
    template <typename E>
    void imul(const E& m);

    //! Divides this matrix with input scalar.
    void idiv(const T& s);

    // MARK: Modifiers

    //! Transposes this matrix.
    void transpose();

    //!
    //! \brief Inverts this matrix.
    //!
    //! This function computes the inverse using Gaussian elimination method.
    //!
    void invert();

    // MARK: Complex getters
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
    //! \warning Should be a square matrix.
    T trace() const;

    //! Returns determinant of this matrix.
    T determinant() const;

    //! Returns diagonal part of this matrix.
    MatrixDiagonal<T, Matrix> diagonal() const;

    //! Returns off-diagonal part of this matrix.
    MatrixDiagonal<T, Matrix> offDiagonal() const;

    //! Returns strictly lower triangle part of this matrix.
    MatrixTriangular<T, Matrix> strictLowerTri() const;

    //! Returns strictly upper triangle part of this matrix.
    MatrixTriangular<T, Matrix> strictUpperTri() const;

    //! Returns lower triangle part of this matrix (including the diagonal).
    MatrixTriangular<T, Matrix> lowerTri() const;

    //! Returns upper triangle part of this matrix (including the diagonal).
    MatrixTriangular<T, Matrix> upperTri() const;

    //! Returns transposed matrix.
    Matrix<T, N, M> transposed() const;

    //! Returns inverse matrix.
    Matrix inverse() const;

    template <typename U>
    MatrixTypeCast<U, Matrix, T> castTo() const;

    // MARK: Setter operators

    //! Assigns input matrix.
    template <typename E>
    Matrix& operator=(const E& m);

    //! Copies to this matrix.
    Matrix& operator=(const Matrix& other);

    //! Addition assignment with input scalar.
    Matrix& operator+=(const T& s);

    //! Addition assignment with input matrix (element-wise).
    template <typename E>
    Matrix& operator+=(const E& m);

    //! Subtraction assignment with input scalar.
    Matrix& operator-=(const T& s);

    //! Subtraction assignment with input matrix (element-wise).
    template <typename E>
    Matrix& operator-=(const E& m);

    //! Multiplication assignment with input scalar.
    Matrix& operator*=(const T& s);

    //! Multiplication assignment with input matrix.
    template <typename E>
    Matrix& operator*=(const E& m);

    //! Division assignment with input scalar.
    Matrix& operator/=(const T& s);

    // MARK: Getter operators

    //! Returns reference of i-th element.
    T& operator[](size_t i);

    //! Returns constant reference of i-th element.
    const T& operator[](size_t i) const;

    //! Returns reference of (i,j) element.
    T& operator()(size_t i, size_t j);

    //! Returns constant reference of (i,j) element.
    const T& operator()(size_t i, size_t j) const;

    //! Returns true if is equal to m.
    template <typename E>
    bool operator==(const MatrixExpression<T, E>& m) const;

    //! Returns true if is not equal to m.
    template <typename E>
    bool operator!=(const MatrixExpression<T, E>& m) const;

    // MARK: Helpers

    //!
    //! \brief Iterates the matrix and invoke given \p func for each index.
    //!
    //! This function iterates the matrix elements and invoke the callback
    //! function \p func. The callback function takes matrix's element as its
    //! input. The order of execution will be the same as the nested for-loop
    //! below:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! for (size_t i = 0; i < mat.rows(); ++i) {
    //!     for (size_t j = 0; j < mat.cols(); ++j) {
    //!         func(mat(i, j));
    //!     }
    //! }
    //! \endcode
    //!
    //! Below is the sample usage:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! mat.forEach([](double elem) {
    //!     printf("%d\n", elem);
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEach(Callback func) const;

    //!
    //! \brief Iterates the matrix and invoke given \p func for each index.
    //!
    //! This function iterates the matrix elements and invoke the callback
    //! function \p func. The callback function takes two parameters which are
    //! the (i, j) indices of the matrix. The order of execution will be the
    //! same as the nested for-loop below:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! for (size_t i = 0; i < mat.rows(); ++i) {
    //!     for (size_t j = 0; j < mat.cols(); ++j) {
    //!         func(i, j);
    //!     }
    //! }
    //! \endcode
    //!
    //! Below is the sample usage:
    //!
    //! \code{.cpp}
    //! MatrixMxN<double> mat(100, 200, 4.0);
    //! mat.forEachIndex([&](size_t i, size_t j) {
    //!     mat(i, j) = 4.0 * i + 7.0 * j + 1.5;
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEachIndex(Callback func) const;

    // MARK: Builders

    //! Makes a M x N matrix with zeros.
    static MatrixConstant<T> makeZero();

    //! Makes a M x N matrix with all diagonal elements to 1, and other elements
    //! to 0.
    static MatrixIdentity<T> makeIdentity();

 private:
    ContainerType _elements;

    template <typename... Params>
    void setRowAt(size_t i, T v, Params... params);
    void setRowAt(size_t i, T v);
};

}  // namespace jet

#include "detail/matrix-inl.h"

#endif  // INCLUDE_JET_MATRIX_H_
