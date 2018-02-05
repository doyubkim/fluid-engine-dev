// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MG_H_
#define INCLUDE_JET_MG_H_

#include <jet/blas.h>

#include <functional>
#include <vector>

namespace jet {

//! Multigrid matrix wrapper.
template <typename BlasType>
struct MgMatrix {
    std::vector<typename BlasType::MatrixType> levels;
    const typename BlasType::MatrixType& operator[](size_t i) const;
    typename BlasType::MatrixType& operator[](size_t i);
    const typename BlasType::MatrixType& finest() const;
    typename BlasType::MatrixType& finest();
};

//! Multigrid vector wrapper.
template <typename BlasType>
struct MgVector {
    std::vector<typename BlasType::VectorType> levels;
    const typename BlasType::VectorType& operator[](size_t i) const;
    typename BlasType::VectorType& operator[](size_t i);
    const typename BlasType::VectorType& finest() const;
    typename BlasType::VectorType& finest();
};

//! Multigrid relax function type.
template <typename BlasType>
using MgRelaxFunc = std::function<void(
    const typename BlasType::MatrixType& A,
    const typename BlasType::VectorType& b, unsigned int numberOfIterations,
    double maxTolerance, typename BlasType::VectorType* x,
    typename BlasType::VectorType* buffer)>;

//! Multigrid restriction function type.
template <typename BlasType>
using MgRestrictFunc =
    std::function<void(const typename BlasType::VectorType& finer,
                       typename BlasType::VectorType* coarser)>;

//! Multigrid correction function type.
template <typename BlasType>
using MgCorrectFunc =
    std::function<void(const typename BlasType::VectorType& coarser,
                       typename BlasType::VectorType* finer)>;

//! Multigrid input parameter set.
template <typename BlasType>
struct MgParameters {
    //! Max number of multigrid levels.
    size_t maxNumberOfLevels = 1;

    //! Number of iteration at restriction step.
    unsigned int numberOfRestrictionIter = 5;

    //! Number of iteration at correction step.
    unsigned int numberOfCorrectionIter = 5;

    //! Number of iteration at coarsest step.
    unsigned int numberOfCoarsestIter = 20;

    //! Number of iteration at final step.
    unsigned int numberOfFinalIter = 20;

    //! Relaxation function such as Jacobi or Gauss-Seidel.
    MgRelaxFunc<BlasType> relaxFunc;

    //! Restrict function that maps finer to coarser grid.
    MgRestrictFunc<BlasType> restrictFunc;

    //! Correction function that maps coarser to finer grid.
    MgCorrectFunc<BlasType> correctFunc;

    //! Max error tolerance.
    double maxTolerance = 1e-9;
};

//! Multigrid result type.
struct MgResult {
    //! Lastly measured norm of residual.
    double lastResidualNorm;
};

//!
//! \brief Performs Multigrid with V-cycle.
//!
//! For given linear system matrix \p A and RHS vector \p b, this function
//! computes the solution \p x using Multigrid method with V-cycle.
//!
template <typename BlasType>
MgResult mgVCycle(const MgMatrix<BlasType>& A, MgParameters<BlasType> params,
                  MgVector<BlasType>* x, MgVector<BlasType>* b,
                  MgVector<BlasType>* buffer);
}  // namespace jet

#include "detail/mg-inl.h"

#endif  // INCLUDE_JET_MG_H_
