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

typedef Array2<double> FdmVector2;
typedef Array2<FdmMatrixRow2> FdmMatrix2;

struct FdmLinearSystem2 {
    FdmMatrix2 A;
    FdmVector2 x, b;
};

struct FdmBlas2 {
    typedef double ScalarType;
    typedef FdmVector2 VectorType;
    typedef FdmMatrix2 MatrixType;

    static void set(double s, FdmVector2* result);

    static void set(const FdmVector2& v, FdmVector2* result);

    static void set(double s, FdmMatrix2* result);

    static void set(const FdmMatrix2& m, FdmMatrix2* result);

    static double dot(const FdmVector2& a, const FdmVector2& b);

    static void axpy(
        double a, const FdmVector2& x, const FdmVector2& y, FdmVector2* result);

    static void mvm(
        const FdmMatrix2& m, const FdmVector2& v, FdmVector2* result);

    static void residual(
        const FdmMatrix2& a,
        const FdmVector2& x,
        const FdmVector2& b,
        FdmVector2* result);

    static double l2Norm(const FdmVector2& v);

    static double lInfNorm(const FdmVector2& v);
};

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM2_H_
