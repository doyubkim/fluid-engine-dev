// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/fdm_mg_solver2.h>

using namespace jet;

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
    _mgParams.relaxFunc = FdmMgUtils2::jacobi;
    _mgParams.restrictFunc = FdmMgUtils2::restrict;
    _mgParams.correctFunc = FdmMgUtils2::correct;
}

const MgParameters<FdmBlas2>& FdmMgSolver2::params() const { return _mgParams; }

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
