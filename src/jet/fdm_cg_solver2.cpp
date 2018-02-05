// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/cg.h>
#include <jet/constants.h>
#include <jet/fdm_cg_solver2.h>

using namespace jet;

FdmCgSolver2::FdmCgSolver2(unsigned int maxNumberOfIterations, double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations),
      _lastNumberOfIterations(0),
      _tolerance(tolerance),
      _lastResidual(kMaxD) {}

bool FdmCgSolver2::solve(FdmLinearSystem2* system) {
    FdmMatrix2& matrix = system->A;
    FdmVector2& solution = system->x;
    FdmVector2& rhs = system->b;

    JET_ASSERT(matrix.size() == rhs.size());
    JET_ASSERT(matrix.size() == solution.size());

    clearCompressedVectors();

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

    cg<FdmBlas2>(matrix, rhs, _maxNumberOfIterations, _tolerance, &solution,
                 &_r, &_d, &_q, &_s, &_lastNumberOfIterations, &_lastResidual);

    return _lastResidual <= _tolerance ||
           _lastNumberOfIterations < _maxNumberOfIterations;
}

bool FdmCgSolver2::solveCompressed(FdmCompressedLinearSystem2* system) {
    MatrixCsrD& matrix = system->A;
    VectorND& solution = system->x;
    VectorND& rhs = system->b;

    clearUncompressedVectors();

    size_t size = solution.size();
    _rComp.resize(size);
    _dComp.resize(size);
    _qComp.resize(size);
    _sComp.resize(size);

    system->x.set(0.0);
    _rComp.set(0.0);
    _dComp.set(0.0);
    _qComp.set(0.0);
    _sComp.set(0.0);

    cg<FdmCompressedBlas2>(matrix, rhs, _maxNumberOfIterations, _tolerance,
                           &solution, &_rComp, &_dComp, &_qComp, &_sComp,
                           &_lastNumberOfIterations, &_lastResidual);

    return _lastResidual <= _tolerance ||
           _lastNumberOfIterations < _maxNumberOfIterations;
}

unsigned int FdmCgSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmCgSolver2::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmCgSolver2::tolerance() const { return _tolerance; }

double FdmCgSolver2::lastResidual() const { return _lastResidual; }

void FdmCgSolver2::clearUncompressedVectors() {
    _r.clear();
    _d.clear();
    _q.clear();
    _s.clear();
}

void FdmCgSolver2::clearCompressedVectors() {
    _rComp.clear();
    _dComp.clear();
    _qComp.clear();
    _sComp.clear();
}
