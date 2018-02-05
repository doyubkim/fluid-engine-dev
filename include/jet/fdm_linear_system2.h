// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_LINEAR_SYSTEM2_H_
#define INCLUDE_JET_FDM_LINEAR_SYSTEM2_H_

#include <jet/array1.h>
#include <jet/array2.h>
#include <jet/matrix_csr.h>
#include <jet/vector_n.h>

namespace jet {

//! The row of FdmMatrix2 where row corresponds to (i, j) grid point.
struct FdmMatrixRow2 {
    //! Diagonal component of the matrix (row, row).
    double center = 0.0;

    //! Off-diagonal element where colum refers to (i+1, j) grid point.
    double right = 0.0;

    //! Off-diagonal element where column refers to (i, j+1) grid point.
    double up = 0.0;
};

//! Vector type for 2-D finite differencing.
typedef Array2<double> FdmVector2;

//! Matrix type for 2-D finite differencing.
typedef Array2<FdmMatrixRow2> FdmMatrix2;

//! Linear system (Ax=b) for 2-D finite differencing.
struct FdmLinearSystem2 {
    //! System matrix.
    FdmMatrix2 A;

    //! Solution vector.
    FdmVector2 x;

    //! RHS vector.
    FdmVector2 b;

    //! Clears all the data.
    void clear();

    //! Resizes the arrays with given grid size.
    void resize(const Size2& size);
};

//! Compressed linear system (Ax=b) for 2-D finite differencing.
struct FdmCompressedLinearSystem2 {
    //! System matrix.
    MatrixCsrD A;

    //! Solution vector.
    VectorND x;

    //! RHS vector.
    VectorND b;

    //! Clears all the data.
    void clear();
};

//! BLAS operator wrapper for 2-D finite differencing.
struct FdmBlas2 {
    typedef double ScalarType;
    typedef FdmVector2 VectorType;
    typedef FdmMatrix2 MatrixType;

    //! Sets entire element of given vector \p result with scalar \p s.
    static void set(ScalarType s, VectorType* result);

    //! Copies entire element of given vector \p result with other vector \p v.
    static void set(const VectorType& v, VectorType* result);

    //! Sets entire element of given matrix \p result with scalar \p s.
    static void set(ScalarType s, MatrixType* result);

    //! Copies entire element of given matrix \p result with other matrix \p v.
    static void set(const MatrixType& m, MatrixType* result);

    //! Performs dot product with vector \p a and \p b.
    static double dot(const VectorType& a, const VectorType& b);

    //! Performs ax + y operation where \p a is a matrix and \p x and \p y are
    //! vectors.
    static void axpy(double a, const VectorType& x, const VectorType& y,
                     VectorType* result);

    //! Performs matrix-vector multiplication.
    static void mvm(const MatrixType& m, const VectorType& v,
                    VectorType* result);

    //! Computes residual vector (b - ax).
    static void residual(const MatrixType& a, const VectorType& x,
                         const VectorType& b, VectorType* result);

    //! Returns L2-norm of the given vector \p v.
    static ScalarType l2Norm(const VectorType& v);

    //! Returns Linf-norm of the given vector \p v.
    static ScalarType lInfNorm(const VectorType& v);
};

//! BLAS operator wrapper for compressed 2-D finite differencing.
struct FdmCompressedBlas2 {
    typedef double ScalarType;
    typedef VectorND VectorType;
    typedef MatrixCsrD MatrixType;

    //! Sets entire element of given vector \p result with scalar \p s.
    static void set(ScalarType s, VectorType* result);

    //! Copies entire element of given vector \p result with other vector \p v.
    static void set(const VectorType& v, VectorType* result);

    //! Sets entire element of given matrix \p result with scalar \p s.
    static void set(ScalarType s, MatrixType* result);

    //! Copies entire element of given matrix \p result with other matrix \p v.
    static void set(const MatrixType& m, MatrixType* result);

    //! Performs dot product with vector \p a and \p b.
    static double dot(const VectorType& a, const VectorType& b);

    //! Performs ax + y operation where \p a is a matrix and \p x and \p y are
    //! vectors.
    static void axpy(double a, const VectorType& x, const VectorType& y,
                     VectorType* result);

    //! Performs matrix-vector multiplication.
    static void mvm(const MatrixType& m, const VectorType& v,
                    VectorType* result);

    //! Computes residual vector (b - ax).
    static void residual(const MatrixType& a, const VectorType& x,
                         const VectorType& b, VectorType* result);

    //! Returns L2-norm of the given vector \p v.
    static ScalarType l2Norm(const VectorType& v);

    //! Returns Linf-norm of the given vector \p v.
    static ScalarType lInfNorm(const VectorType& v);
};

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM2_H_
