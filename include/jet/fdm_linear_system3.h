// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_LINEAR_SYSTEM3_H_
#define INCLUDE_JET_FDM_LINEAR_SYSTEM3_H_

#include <jet/array3.h>

namespace jet {

//! The row of FdmMatrix3 where row corresponds to (i, j, k) grid point.
struct FdmMatrixRow3 {
    //! Diagonal component of the matrix (row, row).
    double center = 0.0;

    //! Off-diagonal element where colum refers to (i+1, j, k) grid point.
    double right = 0.0;

    //! Off-diagonal element where column refers to (i, j+1, k) grid point.
    double up = 0.0;

    //! OFf-diagonal element where column refers to (i, j, k+1) grid point.
    double front = 0.0;
};

//! Vector type for 3-D finite differencing.
typedef Array3<double> FdmVector3;

//! Matrix type for 3-D finite differencing.
typedef Array3<FdmMatrixRow3> FdmMatrix3;

//! Linear system (Ax=b) for 3-D finite differencing.
struct FdmLinearSystem3 {
    FdmMatrix3 A;
    FdmVector3 x, b;
};

//! BLAS operator wrapper for 3-D finite differencing.
struct FdmBlas3 {
    typedef double ScalarType;
    typedef FdmVector3 VectorType;
    typedef FdmMatrix3 MatrixType;

    //! Sets entire element of given vector \p result with scalar \p s.
    static void set(double s, FdmVector3* result);

    //! Copies entire element of given vector \p result with other vector \p v.
    static void set(const FdmVector3& v, FdmVector3* result);

    //! Sets entire element of given matrix \p result with scalar \p s.
    static void set(double s, FdmMatrix3* result);

    //! Copies entire element of given matrix \p result with other matrix \p v.
    static void set(const FdmMatrix3& m, FdmMatrix3* result);

    //! Performs dot product with vector \p a and \p b.
    static double dot(const FdmVector3& a, const FdmVector3& b);

    //! Performs ax + y operation where \p a is a matrix and \p x and \p y are
    //! vectors.
    static void axpy(
        double a, const FdmVector3& x, const FdmVector3& y, FdmVector3* result);

    //! Performs matrix-vector multiplication.
    static void mvm(
        const FdmMatrix3& m, const FdmVector3& v, FdmVector3* result);

    //! Computes residual vector (b - ax).
    static void residual(
        const FdmMatrix3& a,
        const FdmVector3& x,
        const FdmVector3& b,
        FdmVector3* result);

    //! Returns L2-norm of the given vector \p v.
    static double l2Norm(const FdmVector3& v);

    //! Returns Linf-norm of the given vector \p v.
    static double lInfNorm(const FdmVector3& v);
};

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM3_H_
