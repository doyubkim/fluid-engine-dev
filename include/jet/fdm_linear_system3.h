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

typedef Array3<double> FdmVector3;
typedef Array3<FdmMatrixRow3> FdmMatrix3;

struct FdmLinearSystem3 {
    FdmMatrix3 A;
    FdmVector3 x, b;
};

struct FdmBlas3 {
    typedef double ScalarType;
    typedef FdmVector3 VectorType;
    typedef FdmMatrix3 MatrixType;

    static void set(double s, FdmVector3* result);

    static void set(const FdmVector3& v, FdmVector3* result);

    static void set(double s, FdmMatrix3* result);

    static void set(const FdmMatrix3& m, FdmMatrix3* result);

    static double dot(const FdmVector3& a, const FdmVector3& b);

    static void axpy(
        double a, const FdmVector3& x, const FdmVector3& y, FdmVector3* result);

    static void mvm(
        const FdmMatrix3& m, const FdmVector3& v, FdmVector3* result);

    static void residual(
        const FdmMatrix3& a,
        const FdmVector3& x,
        const FdmVector3& b,
        FdmVector3* result);

    static double l2Norm(const FdmVector3& v);

    static double lInfNorm(const FdmVector3& v);
};

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM3_H_
