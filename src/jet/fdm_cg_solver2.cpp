// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constants.h>
#include <jet/cg.h>
#include <jet/fdm_cg_solver2.h>

using namespace jet;

FdmCgSolver2::FdmCgSolver2(
    unsigned int maxNumberOfIterations, double tolerance) :
    _maxNumberOfIterations(maxNumberOfIterations),
    _lastNumberOfIterations(0),
    _tolerance(tolerance),
    _lastResidual(kMaxD) {
}

bool FdmCgSolver2::solve(FdmLinearSystem2* system) {
    FdmMatrix2& matrix = system->A;
    FdmVector2& solution = system->x;
    FdmVector2& rhs = system->b;

    JET_ASSERT(matrix.size() == rhs.size());
    JET_ASSERT(matrix.size() == solution.size());

    Size2 size = matrix.size();
    _r.resize(size);
    _d.resize(size);
    _q.resize(size);
    _s.resize(size);

    system->x.set(0.0);
    _r.set(0.0);
    _d.set(0.0);
    _q.set(0.0);
    _s.set(0.0);

    cg<FdmBlas2>(
        matrix,
        rhs,
        _maxNumberOfIterations,
        _tolerance,
        &solution,
        &_r,
        &_d,
        &_q,
        &_s,
        &_lastNumberOfIterations,
        &_lastResidual);

    return _lastResidual <= _tolerance
        || _lastNumberOfIterations < _maxNumberOfIterations;
}

unsigned int FdmCgSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmCgSolver2::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmCgSolver2::tolerance() const {
    return _tolerance;
}

double FdmCgSolver2::lastResidual() const {
    return _lastResidual;
}
