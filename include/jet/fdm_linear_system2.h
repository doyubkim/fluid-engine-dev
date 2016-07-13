// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_LINEAR_SYSTEM2_H_
#define INCLUDE_JET_FDM_LINEAR_SYSTEM2_H_

#include <jet/array2.h>

namespace jet {

//! The row of FdmMatrix2 where row corresponds to (i, j, k) grid point.
struct FdmMatrixRow2 {
    //! Diagonal component of the matrix (row, row).
    double center = 0.0;

    //! Off-diagonal element where colum refers to (i+1, j, k) grid point.
    double right = 0.0;

    //! Off-diagonal element where column refers to (i, j+1, k) grid point.
    double up = 0.0;
};

//! Vector type for 2-D finite differencing.
typedef Array2<double> FdmVector2;

//! Matrix type for 2-D finite differencing.
typedef Array2<FdmMatrixRow2> FdmMatrix2;

//! Linear system (Ax=b) for 2-D finite differencing.
struct FdmLinearSystem2 {
    FdmMatrix2 A;
    FdmVector2 x, b;
};

//! BLAS operator wrapper for 2-D finite differencing.
struct FdmBlas2 {
    typedef double ScalarType;
    typedef FdmVector2 VectorType;
    typedef FdmMatrix2 MatrixType;

    //! Sets entire element of given vector \p result with scalar \p s.
    static void set(double s, FdmVector2* result);

    //! Copies entire element of given vector \p result with other vector \p v.
    static void set(const FdmVector2& v, FdmVector2* result);

    //! Sets entire element of given matrix \p result with scalar \p s.
    static void set(double s, FdmMatrix2* result);

    //! Copies entire element of given matrix \p result with other matrix \p v.
    static void set(const FdmMatrix2& m, FdmMatrix2* result);

    //! Performs dot product with vector \p a and \p b.
    static double dot(const FdmVector2& a, const FdmVector2& b);

    //! Performs ax + y operation where \p a is a matrix and \p x and \p y are
    //! vectors.
    static void axpy(
        double a, const FdmVector2& x, const FdmVector2& y, FdmVector2* result);

    //! Performs matrix-vector multiplication.
    static void mvm(
        const FdmMatrix2& m, const FdmVector2& v, FdmVector2* result);

    //! Computes residual vector (b - ax).
    static void residual(
        const FdmMatrix2& a,
        const FdmVector2& x,
        const FdmVector2& b,
        FdmVector2* result);

    //! Returns L2-norm of the given vector \p v.
    static double l2Norm(const FdmVector2& v);

    //! Returns Linf-norm of the given vector \p v.
    static double lInfNorm(const FdmVector2& v);
};

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM2_H_
