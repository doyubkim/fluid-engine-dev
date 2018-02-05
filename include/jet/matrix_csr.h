// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_CSR_H_
#define INCLUDE_JET_MATRIX_CSR_H_

#include <jet/matrix_expression.h>
#include <jet/size2.h>
#include <jet/vector_n.h>

#include <tuple>

namespace jet {

template <typename T>
class MatrixCsr;

//!
//! \brief Vector expression for CSR matrix-vector multiplication.
//!
//! This vector expression represents a CSR matrix-vector operation that
//! takes one CSR input matrix expression and one vector expression.
//!
//! \tparam T   Element value type.
//! \tparam VE  Vector expression.
//!
template <typename T, typename VE>
class MatrixCsrVectorMul
    : public VectorExpression<T, MatrixCsrVectorMul<T, VE>> {
 public:
    MatrixCsrVectorMul(const MatrixCsr<T>& m, const VE& v);
    MatrixCsrVectorMul(const MatrixCsrVectorMul&);

    //! Size of the vector.
    size_t size() const;

    //! Returns vector element at i.
    T operator[](size_t i) const;

 private:
    const MatrixCsr<T>& _m;
    const VE& _v;
    const VE* _v2;
};

//!
//! \brief Matrix expression for CSR matrix-matrix multiplication.
//!
//! This matrix expression represents a CSR matrix-matrix operation that
//! takes one CSR input matrix expression and one (probably dense) matrix
//! expression.
//!
//! \tparam T   Element value type.
//! \tparam ME  Matrix expression.
//!
template <typename T, typename ME>
class MatrixCsrMatrixMul
    : public MatrixExpression<T, MatrixCsrMatrixMul<T, ME>> {
 public:
    MatrixCsrMatrixMul(const MatrixCsr<T>& m1, const ME& m2);

    //! Size of the matrix.
    Size2 size() const;

    //! Number of rows.
    size_t rows() const;

    //! Number of columns.
    size_t cols() const;

    //! Returns matrix element at (i, j).
    T operator()(size_t i, size_t j) const;

 private:
    const MatrixCsr<T>& _m1;
    const ME& _m2;
    const T* const _nnz;
    const size_t* const _rp;
    const size_t* const _ci;
};

//!
//! \brief Compressed Sparse Row (CSR) matrix class.
//!
//! This class defines Compressed Sparse Row (CSR) matrix using arrays of
//! non-zero elements, row pointers, and column indices.
//!
//! \see http://www.netlib.org/utk/people/JackDongarra/etemplates/node373.html
//!
//! \tparam T Type of the element.
//!
template <typename T>
class MatrixCsr final : public MatrixExpression<T, MatrixCsr<T>> {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "MatrixCsr only can be instantiated with floating point types");

    struct Element {
        size_t i;
        size_t j;
        T value;

        Element();

        Element(size_t i, size_t j, const T& value);
    };

    typedef std::vector<T> NonZeroContainerType;
    typedef typename NonZeroContainerType::iterator NonZeroIterator;
    typedef typename NonZeroContainerType::const_iterator ConstNonZeroIterator;

    typedef std::vector<size_t> IndexContainerType;
    typedef typename IndexContainerType::iterator IndexIterator;
    typedef typename IndexContainerType::const_iterator ConstIndexIterator;

    // MARK: Constructors

    //! Constructs an empty matrix.
    MatrixCsr();

    //!
    //! \brief Compresses given initializer list \p lst into a sparse matrix.
    //!
    //! This constructor will build a matrix with given initializer list \p lst
    //! such as
    //!
    //! \code{.cpp}
    //! MatrixCsr<float> mat = {
    //!     {1.f, 0.f, 0.f, 3.f},
    //!     {0.f, 3.f, 5.f, 1.f},
    //!     {4.f, 0.f, 1.f, 5.f}
    //! };
    //! \endcode
    //!
    //! Note the initializer has 4x3 structure which will create 4x3 matrix.
    //! During the process, zero elements (less than \p epsilon) will not be
    //! stored.
    //!
    //! \param lst Initializer list that should be copy to the new matrix.
    //!
    MatrixCsr(const std::initializer_list<std::initializer_list<T>>& lst,
              T epsilon = std::numeric_limits<T>::epsilon());

    //!
    //! \brief Compresses input (dense) matrix expression into a sparse matrix.
    //!
    //! This function sets this sparse matrix with dense input matrix.
    //! During the process, zero elements (less than \p epsilon) will not be
    //! stored.
    //!
    template <typename E>
    MatrixCsr(const MatrixExpression<T, E>& other,
              T epsilon = std::numeric_limits<T>::epsilon());

    //! Copy constructor.
    MatrixCsr(const MatrixCsr& other);

    //! Move constructor.
    MatrixCsr(MatrixCsr&& other);

    // MARK: Basic setters

    //! Clears the matrix and make it zero-dimensional.
    void clear();

    //! Sets whole matrix with input scalar.
    void set(const T& s);

    //! Copy from given sparse matrix.
    void set(const MatrixCsr& other);

    //! Reserves memory space of this matrix.
    void reserve(size_t rows, size_t cols, size_t numNonZeros);

    //!
    //! \brief Compresses given initializer list \p lst into a sparse matrix.
    //!
    //! This function will fill the matrix with given initializer list \p lst
    //! such as
    //!
    //! \code{.cpp}
    //! MatrixCsr<float> mat;
    //! mat.compress({
    //!     {1.f, 0.f, 0.f, 3.f},
    //!     {0.f, 3.f, 5.f, 1.f},
    //!     {4.f, 0.f, 1.f, 5.f}
    //! });
    //! \endcode
    //!
    //! Note the initializer has 4x3 structure which will resize to 4x3 matrix.
    //! During the process, zero elements (less than \p epsilon) will not be
    //! stored.
    //!
    //! \param lst Initializer list that should be copy to the new matrix.
    //!
    void compress(const std::initializer_list<std::initializer_list<T>>& lst,
                  T epsilon = std::numeric_limits<T>::epsilon());

    //!
    //! \brief Compresses input (dense) matrix expression into a sparse matrix.
    //!
    //! This function sets this sparse matrix with dense input matrix.
    //! During the process, zero elements (less than \p epsilon) will not be
    //! stored.
    //!
    template <typename E>
    void compress(const MatrixExpression<T, E>& other,
                  T epsilon = std::numeric_limits<T>::epsilon());

    //! Adds non-zero element to (i, j).
    void addElement(size_t i, size_t j, const T& value);

    //! Adds non-zero element.
    void addElement(const Element& element);

    //!
    //! Adds a row to the sparse matrix.
    //!
    //! \param nonZeros - Array of non-zero elements for the row.
    //! \param columnIndices - Array of column indices for the row.
    //!
    void addRow(const NonZeroContainerType& nonZeros,
                const IndexContainerType& columnIndices);

    //! Sets non-zero element to (i, j).
    void setElement(size_t i, size_t j, const T& value);

    //! Sets non-zero element.
    void setElement(const Element& element);

    // MARK: Basic getters
    bool isEqual(const MatrixCsr& other) const;

    //! Returns true if this matrix is similar to the input matrix within the
    //! given tolerance.
    bool isSimilar(const MatrixCsr& other,
                   double tol = std::numeric_limits<double>::epsilon()) const;

    //! Returns true if this matrix is a square matrix.
    bool isSquare() const;

    //! Returns the size of this matrix.
    Size2 size() const;

    //! Returns number of rows of this matrix.
    size_t rows() const;

    //! Returns number of columns of this matrix.
    size_t cols() const;

    //! Returns the number of non-zero elements.
    size_t numberOfNonZeros() const;

    //! Returns i-th non-zero element.
    const T& nonZero(size_t i) const;

    //! Returns i-th non-zero element.
    T& nonZero(size_t i);

    //! Returns i-th row pointer.
    const size_t& rowPointer(size_t i) const;

    //! Returns i-th column index.
    const size_t& columnIndex(size_t i) const;

    //! Returns pointer of the non-zero elements data.
    T* nonZeroData();

    //! Returns constant pointer of the non-zero elements data.
    const T* const nonZeroData() const;

    //! Returns constant pointer of the row pointers data.
    const size_t* const rowPointersData() const;

    //! Returns constant pointer of the column indices data.
    const size_t* const columnIndicesData() const;

    //! Returns the begin iterator of the non-zero elements.
    NonZeroIterator nonZeroBegin();

    //! Returns the begin const iterator of the non-zero elements.
    ConstNonZeroIterator nonZeroBegin() const;

    //! Returns the end iterator of the non-zero elements.
    NonZeroIterator nonZeroEnd();

    //! Returns the end const iterator of the non-zero elements.
    ConstNonZeroIterator nonZeroEnd() const;

    //! Returns the begin iterator of the row pointers.
    IndexIterator rowPointersBegin();

    //! Returns the begin const iterator of the row pointers.
    ConstIndexIterator rowPointersBegin() const;

    //! Returns the end iterator of the row pointers.
    IndexIterator rowPointersEnd();

    //! Returns the end const iterator of the row pointers.
    ConstIndexIterator rowPointersEnd() const;

    //! Returns the begin iterator of the column indices.
    IndexIterator columnIndicesBegin();

    //! Returns the begin const iterator of the column indices.
    ConstIndexIterator columnIndicesBegin() const;

    //! Returns the end iterator of the column indices.
    IndexIterator columnIndicesEnd();

    //! Returns the end const iterator of the column indices.
    ConstIndexIterator columnIndicesEnd() const;

    // MARK: Binary operator methods - new instance = this instance (+) input

    //! Returns this matrix + input scalar.
    MatrixCsr add(const T& s) const;

    //! Returns this matrix + input matrix (element-wise).
    MatrixCsr add(const MatrixCsr& m) const;

    //! Returns this matrix - input scalar.
    MatrixCsr sub(const T& s) const;

    //! Returns this matrix - input matrix (element-wise).
    MatrixCsr sub(const MatrixCsr& m) const;

    //! Returns this matrix * input scalar.
    MatrixCsr mul(const T& s) const;

    //! Returns this matrix * input vector.
    template <typename VE>
    MatrixCsrVectorMul<T, VE> mul(const VectorExpression<T, VE>& v) const;

    //! Returns this matrix * input matrix.
    template <typename ME>
    MatrixCsrMatrixMul<T, ME> mul(const MatrixExpression<T, ME>& m) const;

    //! Returns this matrix / input scalar.
    MatrixCsr div(const T& s) const;

    // MARK: Binary operator methods - new instance = input (+) this instance

    //! Returns input scalar + this matrix.
    MatrixCsr radd(const T& s) const;

    //! Returns input matrix + this matrix (element-wise).
    MatrixCsr radd(const MatrixCsr& m) const;

    //! Returns input scalar - this matrix.
    MatrixCsr rsub(const T& s) const;

    //! Returns input matrix - this matrix (element-wise).
    MatrixCsr rsub(const MatrixCsr& m) const;

    //! Returns input scalar * this matrix.
    MatrixCsr rmul(const T& s) const;

    //! Returns input matrix / this scalar.
    MatrixCsr rdiv(const T& s) const;

    // MARK: Augmented operator methods - this instance (+)= input

    //! Adds input scalar to this matrix.
    void iadd(const T& s);

    //! Adds input matrix to this matrix (element-wise).
    void iadd(const MatrixCsr& m);

    //! Subtracts input scalar from this matrix.
    void isub(const T& s);

    //! Subtracts input matrix from this matrix (element-wise).
    void isub(const MatrixCsr& m);

    //! Multiplies input scalar to this matrix.
    void imul(const T& s);

    //! Multiplies input matrix to this matrix.
    template <typename ME>
    void imul(const MatrixExpression<T, ME>& m);

    //! Divides this matrix with input scalar.
    void idiv(const T& s);

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

    //! Type-casts to different value-typed matrix.
    template <typename U>
    MatrixCsr<U> castTo() const;

    // MARK: Setter operators

    //!
    //! \brief Compresses input (dense) matrix expression into a sparse matrix.
    //!
    //! This function sets this sparse matrix with dense input matrix.
    //! During the process, zero elements (less than \p epsilon) will not be
    //! stored.
    //!
    template <typename E>
    MatrixCsr& operator=(const E& m);

    //! Copies to this matrix.
    MatrixCsr& operator=(const MatrixCsr& other);

    //! Moves to this matrix.
    MatrixCsr& operator=(MatrixCsr&& other);

    //! Addition assignment with input scalar.
    MatrixCsr& operator+=(const T& s);

    //! Addition assignment with input matrix (element-wise).
    MatrixCsr& operator+=(const MatrixCsr& m);

    //! Subtraction assignment with input scalar.
    MatrixCsr& operator-=(const T& s);

    //! Subtraction assignment with input matrix (element-wise).
    MatrixCsr& operator-=(const MatrixCsr& m);

    //! Multiplication assignment with input scalar.
    MatrixCsr& operator*=(const T& s);

    //! Multiplication assignment with input matrix.
    template <typename ME>
    MatrixCsr& operator*=(const MatrixExpression<T, ME>& m);

    //! Division assignment with input scalar.
    MatrixCsr& operator/=(const T& s);

    // MARK: Getter operators

    //! Returns (i,j) element.
    T operator()(size_t i, size_t j) const;

    //! Returns true if is equal to m.
    bool operator==(const MatrixCsr& m) const;

    //! Returns true if is not equal to m.
    bool operator!=(const MatrixCsr& m) const;

    // MARK: Builders
    //! Makes a m x m matrix with all diagonal elements to 1, and other elements
    //! to 0.
    static MatrixCsr<T> makeIdentity(size_t m);

 private:
    Size2 _size;
    NonZeroContainerType _nonZeros;
    IndexContainerType _rowPointers;
    IndexContainerType _columnIndices;

    size_t hasElement(size_t i, size_t j) const;

    template <typename Op>
    MatrixCsr binaryOp(const MatrixCsr& m, Op op) const;
};

//! Float-type CSR matrix.
typedef MatrixCsr<float> MatrixCsrF;

//! Double-type CSR matrix.
typedef MatrixCsr<double> MatrixCsrD;

}  // namespace jet

#include "detail/matrix_csr-inl.h"

#endif  // INCLUDE_JET_MATRIX_CSR_H_
