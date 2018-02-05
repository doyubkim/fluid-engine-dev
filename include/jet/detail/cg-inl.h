// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_CG_INL_H_
#define INCLUDE_JET_DETAIL_CG_INL_H_

#include <jet/constants.h>
#include <limits>

namespace jet {

template <
    typename BlasType,
    typename PrecondType>
void pcg(
    const typename BlasType::MatrixType& A,
    const typename BlasType::VectorType& b,
    unsigned int maxNumberOfIterations,
    double tolerance,
    PrecondType* M,
    typename BlasType::VectorType* x,
    typename BlasType::VectorType* r,
    typename BlasType::VectorType* d,
    typename BlasType::VectorType* q,
    typename BlasType::VectorType* s,
    unsigned int* lastNumberOfIterations,
    double* lastResidualNorm) {
    // Clear
    BlasType::set(0, r);
    BlasType::set(0, d);
    BlasType::set(0, q);
    BlasType::set(0, s);

    // r = b - Ax
    BlasType::residual(A, *x, b, r);

    // d = M^-1r
    M->solve(*r, d);

    // sigmaNew = r.d
    double sigmaNew = BlasType::dot(*r, *d);

    unsigned int iter = 0;
    bool trigger = false;
    while (sigmaNew > square(tolerance) && iter < maxNumberOfIterations) {
        // q = Ad
        BlasType::mvm(A, *d, q);

        // alpha = sigmaNew/d.q
        double alpha = sigmaNew / BlasType::dot(*d, *q);

        // x = x + alpha*d
        BlasType::axpy(alpha, *d, *x, x);

        // if i is divisible by 50...
        if (trigger || (iter % 50 == 0 && iter > 0)) {
            // r = b - Ax
            BlasType::residual(A, *x, b, r);
            trigger = false;
        } else {
            // r = r - alpha*q
            BlasType::axpy(-alpha, *q, *r, r);
        }

        // s = M^-1r
        M->solve(*r, s);

        // sigmaOld = sigmaNew
        double sigmaOld = sigmaNew;

        // sigmaNew = r.s
        sigmaNew = BlasType::dot(*r, *s);

        if (sigmaNew > sigmaOld) {
            trigger = true;
        }

        // beta = sigmaNew/sigmaOld
        double beta = sigmaNew / sigmaOld;

        // d = s + beta*d
        BlasType::axpy(beta, *d, *s, d);

        ++iter;
    }

    *lastNumberOfIterations = iter;

    // std::fabs(sigmaNew) - Workaround for negative zero
    *lastResidualNorm = std::sqrt(std::fabs(sigmaNew));
}

template <typename BlasType>
void cg(
    const typename BlasType::MatrixType& A,
    const typename BlasType::VectorType& b,
    unsigned int maxNumberOfIterations,
    double tolerance,
    typename BlasType::VectorType* x,
    typename BlasType::VectorType* r,
    typename BlasType::VectorType* d,
    typename BlasType::VectorType* q,
    typename BlasType::VectorType* s,
    unsigned int* lastNumberOfIterations,
    double* lastResidualNorm) {
    typedef NullCgPreconditioner<BlasType> PrecondType;
    PrecondType precond;
    pcg<BlasType, PrecondType>(
        A,
        b,
        maxNumberOfIterations,
        tolerance,
        &precond,
        x,
        r,
        d,
        q,
        s,
        lastNumberOfIterations,
        lastResidualNorm);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CG_INL_H_
