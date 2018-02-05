// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/fdm_gauss_seidel_solver3.h>
#include <jet/fdm_mg_solver3.h>

using namespace jet;

FdmMgSolver3::FdmMgSolver3(size_t maxNumberOfLevels,
                           unsigned int numberOfRestrictionIter,
                           unsigned int numberOfCorrectionIter,
                           unsigned int numberOfCoarsestIter,
                           unsigned int numberOfFinalIter, double maxTolerance,
                           double sorFactor, bool useRedBlackOrdering) {
    _mgParams.maxNumberOfLevels = maxNumberOfLevels;
    _mgParams.numberOfRestrictionIter = numberOfRestrictionIter;
    _mgParams.numberOfCorrectionIter = numberOfCorrectionIter;
    _mgParams.numberOfCoarsestIter = numberOfCoarsestIter;
    _mgParams.numberOfFinalIter = numberOfFinalIter;
    _mgParams.maxTolerance = maxTolerance;
    if (useRedBlackOrdering) {
        _mgParams.relaxFunc = [sorFactor](
            const FdmMatrix3& A, const FdmVector3& b,
            unsigned int numberOfIterations, double maxTolerance, FdmVector3* x,
            FdmVector3* buffer) {
            UNUSED_VARIABLE(buffer);
            UNUSED_VARIABLE(maxTolerance);

            for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
                FdmGaussSeidelSolver3::relaxRedBlack(A, b, sorFactor, x);
            }
        };
    } else {
        _mgParams.relaxFunc = [sorFactor](
            const FdmMatrix3& A, const FdmVector3& b,
            unsigned int numberOfIterations, double maxTolerance, FdmVector3* x,
            FdmVector3* buffer) {
            UNUSED_VARIABLE(buffer);
            UNUSED_VARIABLE(maxTolerance);

            for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
                FdmGaussSeidelSolver3::relax(A, b, sorFactor, x);
            }
        };
    }
    _mgParams.restrictFunc = FdmMgUtils3::restrict;
    _mgParams.correctFunc = FdmMgUtils3::correct;

    _sorFactor = sorFactor;
    _useRedBlackOrdering = useRedBlackOrdering;
}

const MgParameters<FdmBlas3>& FdmMgSolver3::params() const { return _mgParams; }

double FdmMgSolver3::sorFactor() const { return _sorFactor; }

bool FdmMgSolver3::useRedBlackOrdering() const { return _useRedBlackOrdering; }

bool FdmMgSolver3::solve(FdmLinearSystem3* system) {
    UNUSED_VARIABLE(system);
    return false;
}

bool FdmMgSolver3::solve(FdmMgLinearSystem3* system) {
    FdmMgVector3 buffer = system->x;
    auto result =
        mgVCycle(system->A, _mgParams, &system->x, &system->b, &buffer);
    return result.lastResidualNorm < _mgParams.maxTolerance;
}
