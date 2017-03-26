// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_MG_INL_H_
#define INCLUDE_JET_DETAIL_MG_INL_H_

#include <jet/mg.h>

namespace jet {

namespace internal {

template <typename BlasType>
MgResult mgVCycle(const MgMatrix<BlasType>& A, MgParameters<BlasType> params,
                  unsigned int currentLevel, MgVector<BlasType>* x,
                  MgVector<BlasType>* b, MgVector<BlasType>* buffer) {
    // 1) Relax a few times on Ax = b, with arbitrary x
    params.relaxFunc(A[currentLevel], (*b)[currentLevel],
                     params.numberOfRestrictionIter, params.maxTolerance,
                     &((*x)[currentLevel]), &((*buffer)[currentLevel]));

    // 2) if currentLevel is the coarsest grid, goto 5)
    if (currentLevel < params.maxNumberOfLevels - 1) {
        auto r = buffer;
        BlasType::residual(A[currentLevel], (*x)[currentLevel],
                           (*b)[currentLevel], &(*r)[currentLevel]);
        params.restrictFunc((*r)[currentLevel], &(*b)[currentLevel + 1]);

        BlasType::set(0.0, &(*x)[currentLevel + 1]);

        params.maxTolerance *= 0.5;
        // Solve Ae = r
        mgVCycle(A, params, currentLevel + 1, x, b, buffer);
        params.maxTolerance *= 2.0;

        // 3) correct
        params.correctFunc((*x)[currentLevel + 1], &(*x)[currentLevel]);

        // 4) relax nItr times on Ax = b, with initial guess x
        if (currentLevel > 0) {
            params.relaxFunc(A[currentLevel], (*b)[currentLevel],
                             params.numberOfCorrectionIter, params.maxTolerance,
                             &((*x)[currentLevel]), &((*buffer)[currentLevel]));
        } else {
            params.relaxFunc(A[currentLevel], (*b)[currentLevel],
                             params.numberOfFinalIter, params.maxTolerance,
                             &((*x)[currentLevel]), &((*buffer)[currentLevel]));
        }
    } else {
        // 5) solve directly with initial guess x
        params.relaxFunc(A[currentLevel], (*b)[currentLevel],
                         params.numberOfCoarsestIter, params.maxTolerance,
                         &((*x)[currentLevel]), &((*buffer)[currentLevel]));

        BlasType::residual(A[currentLevel], (*x)[currentLevel],
                           (*b)[currentLevel], &(*buffer)[currentLevel]);
    }

    BlasType::residual(A[currentLevel], (*x)[currentLevel], (*b)[currentLevel],
                       &(*buffer)[currentLevel]);

    MgResult result;
    result.lastResidualNorm = BlasType::l2Norm((*buffer)[currentLevel]);
    return result;
}

}  // namespace internal

template <typename BlasType>
const typename BlasType::MatrixType& MgMatrix<BlasType>::operator[](
    size_t i) const {
    return levels[i];
}

template <typename BlasType>
typename BlasType::MatrixType& MgMatrix<BlasType>::operator[](size_t i) {
    return levels[i];
}

template <typename BlasType>
const typename BlasType::VectorType& MgVector<BlasType>::operator[](
    size_t i) const {
    return levels[i];
}

template <typename BlasType>
typename BlasType::VectorType& MgVector<BlasType>::operator[](size_t i) {
    return levels[i];
}

template <typename BlasType>
MgResult mgVCycle(const MgMatrix<BlasType>& A, MgParameters<BlasType> params,
                  MgVector<BlasType>* x, MgVector<BlasType>* b,
                  MgVector<BlasType>* buffer) {
    return internal::mgVCycle<BlasType>(A, params, 0u, x, b, buffer);
}
}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_MG_INL_H_
