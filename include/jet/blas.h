// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_BLAS_H_
#define INCLUDE_JET_BLAS_H_

#include <jet/vector4.h>
#include <jet/matrix4x4.h>

namespace jet {

template <typename S, typename V, typename M>
struct Blas {
    typedef S ScalarType;
    typedef V VectorType;
    typedef M MatrixType;

    static void set(ScalarType s, VectorType* result);

    static void set(const VectorType& v, VectorType* result);

    static void set(ScalarType s, MatrixType* result);

    static void set(const MatrixType& m, MatrixType* result);

    static ScalarType dot(const VectorType& a, const VectorType& b);

    static void axpy(
        ScalarType a,
        const VectorType& x,
        const VectorType& y,
        VectorType* result);

    static void mvm(
        const MatrixType& m,
        const VectorType& v,
        VectorType* result);

    static void residual(
        const MatrixType& a,
        const VectorType& x,
        const VectorType& b,
        VectorType* result);

    static ScalarType l2Norm(const VectorType& v);

    static ScalarType lInfNorm(const VectorType& v);
};

}  // namespace jet

#include "detail/blas-inl.h"

#endif  // INCLUDE_JET_BLAS_H_
