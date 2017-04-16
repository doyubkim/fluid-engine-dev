// Copyright (c) 2017 Doyub Kim
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

template <typename BlasType>
struct MgMatrix {
    std::vector<typename BlasType::MatrixType> levels;
    const typename BlasType::MatrixType& operator[](size_t i) const;
    typename BlasType::MatrixType& operator[](size_t i);
};

template <typename BlasType>
struct MgVector {
    std::vector<typename BlasType::VectorType> levels;
    const typename BlasType::VectorType& operator[](size_t i) const;
    typename BlasType::VectorType& operator[](size_t i);
};

template <typename BlasType>
using MgRelaxFunc = std::function<void(
    const typename BlasType::MatrixType& A,
    const typename BlasType::VectorType& b, unsigned int numberOfIterations,
    double maxTolerance, typename BlasType::VectorType* x,
    typename BlasType::VectorType* buffer)>;

template <typename BlasType>
using MgRestrictFunc =
    std::function<void(const typename BlasType::VectorType& finer,
                       typename BlasType::VectorType* coarser)>;

template <typename BlasType>
using MgCorrectFunc =
    std::function<void(const typename BlasType::VectorType& coarser,
                       typename BlasType::VectorType* finer)>;

template <typename BlasType>
struct MgParameters {
    unsigned int maxNumberOfLevels = 1;
    unsigned int numberOfRestrictionIter = 10;
    unsigned int numberOfCorrectionIter = 10;
    unsigned int numberOfCoarsestIter = 10;
    unsigned int numberOfFinalIter = 10;
    MgRelaxFunc<BlasType> relaxFunc;
    MgRestrictFunc<BlasType> restrictFunc;
    MgCorrectFunc<BlasType> correctFunc;
    double maxTolerance = 1e-9;
};

struct MgResult {
    double lastResidualNorm;
};

template <typename BlasType>
MgResult mgVCycle(const MgMatrix<BlasType>& A, MgParameters<BlasType> params,
                  MgVector<BlasType>* x, MgVector<BlasType>* b,
                  MgVector<BlasType>* buffer);
}  // namespace jet

#include "detail/mg-inl.h"

#endif  // INCLUDE_JET_MG_H_
