// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CG_H_
#define INCLUDE_JET_CG_H_

#include <jet/blas.h>

namespace jet {

//!
//! \brief No-op preconditioner for conjugate gradient.
//!
//! This preconditioner does nothing but simply copies the input vector to the
//! output vector. Thus, it can be considered as an identity matrix.
//!
template <typename BlasType>
struct NullCgPreconditioner final {
    void build(const typename BlasType::MatrixType&) {}

    void solve(
        const typename BlasType::VectorType& b,
        typename BlasType::VectorType* x) {
        BlasType::set(b, x);
    }
};

//!
//! \brief Solves conjugate gradient.
//!
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
    double* lastResidualNorm);

//!
//! \brief Solves pre-conditioned conjugate gradient.
//!
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
    double* lastResidualNorm);

}  // namespace jet

#include "detail/cg-inl.h"

#endif  // INCLUDE_JET_CG_H_
