// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/fdm_mg_solver2.h>

using namespace jet;

namespace {

void jacobi(const FdmMatrix2& A, const FdmVector2& b,
            unsigned int numberOfIterations, double maxTolerance, FdmVector2* x,
            FdmVector2* xTemp) {
    UNUSED_VARIABLE(maxTolerance);

    Size2 size = A.size();

    for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
        A.parallelForEachIndex([&](size_t i, size_t j) {
            double r =
                ((i > 0) ? A(i - 1, j).right * (*x)(i - 1, j) : 0.0) +
                ((i + 1 < size.x) ? A(i, j).right * (*x)(i + 1, j) : 0.0) +
                ((j > 0) ? A(i, j - 1).up * (*x)(i, j - 1) : 0.0) +
                ((j + 1 < size.y) ? A(i, j).up * (*x)(i, j + 1) : 0.0);

            (*xTemp)(i, j) = (b(i, j) - r) / A(i, j).center;
        });

        x->swap(*xTemp);
    }

    FdmBlas2::residual(A, *x, b, xTemp);
}
}

FdmMgSolver2::FdmMgSolver2(size_t maxNumberOfLevels,
                           unsigned int numberOfRestrictionIter,
                           unsigned int numberOfCorrectionIter,
                           unsigned int numberOfCoarsestIter,
                           unsigned int numberOfFinalIter, double maxTolerance)

{
    _mgParams.maxNumberOfLevels = maxNumberOfLevels;
    _mgParams.numberOfRestrictionIter = numberOfRestrictionIter;
    _mgParams.numberOfCorrectionIter = numberOfCorrectionIter;
    _mgParams.numberOfCoarsestIter = numberOfCoarsestIter;
    _mgParams.numberOfFinalIter = numberOfFinalIter;
    _mgParams.maxTolerance = maxTolerance;
    _mgParams.relaxFunc = jacobi;
    _mgParams.restrictFunc = FdmMgUtils2::restrict;
    _mgParams.correctFunc = FdmMgUtils2::correct;
}

bool FdmMgSolver2::solve(FdmLinearSystem2* system) {
    UNUSED_VARIABLE(system);
    return false;
}

bool FdmMgSolver2::solve(FdmMgLinearSystem2* system) {
    FdmMgVector2 buffer = system->x;
    auto result =
        mgVCycle(system->A, _mgParams, &system->x, &system->b, &buffer);
    return result.lastResidualNorm < _mgParams.maxTolerance;
}
