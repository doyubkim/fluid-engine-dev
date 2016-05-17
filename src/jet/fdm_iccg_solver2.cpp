// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constants.h>
#include <jet/cg.h>
#include <jet/fdm_iccg_solver2.h>

using namespace jet;

void FdmIccgSolver2::Preconditioner::build(const FdmMatrix2& matrix) {
    Size2 size = matrix.size();
    A = matrix.constAccessor();

    d.resize(size, 0.0);
    y.resize(size, 0.0);

    matrix.forEachIndex([&](size_t i, size_t j) {
        double denom
            = matrix(i, j).center
            - ((i > 0) ?
                square(matrix(i - 1, j).right) * d(i - 1, j) : 0.0)
            - ((j > 0) ?
                square(matrix(i, j - 1).up)    * d(i, j - 1) : 0.0);

        if (std::fabs(denom) > 0.0) {
            d(i, j) = 1.0 / denom;
        } else {
            d(i, j) = 0.0;
        }
    });
}

void FdmIccgSolver2::Preconditioner::solve(
    const FdmVector2& b,
    FdmVector2* x) {
    Size2 size = b.size();
    ssize_t sx = static_cast<ssize_t>(size.x);
    ssize_t sy = static_cast<ssize_t>(size.y);

    b.forEachIndex([&](size_t i, size_t j) {
        y(i, j)
            = (b(i, j)
            - ((i > 0) ? A(i - 1, j).right * y(i - 1, j) : 0.0)
            - ((j > 0) ? A(i, j - 1).up    * y(i, j - 1) : 0.0))
            * d(i, j);
    });

    for (ssize_t j = sy - 1; j >= 0; --j) {
        for (ssize_t i = sx - 1; i >= 0; --i) {
            (*x)(i, j)
                = (y(i, j)
                - ((i + 1 < sx) ? A(i, j).right * (*x)(i + 1, j) : 0.0)
                - ((j + 1 < sy) ? A(i, j).up    * (*x)(i, j + 1) : 0.0))
                * d(i, j);
        }
    }
}

FdmIccgSolver2::FdmIccgSolver2(
    unsigned int maxNumberOfIterations,
    double tolerance) :
    _maxNumberOfIterations(maxNumberOfIterations),
    _lastNumberOfIterations(0),
    _tolerance(tolerance),
    _lastResidualNorm(kMaxD) {
}

bool FdmIccgSolver2::solve(FdmLinearSystem2* system) {
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

    _precond.build(matrix);

    pcg<FdmBlas2, Preconditioner>(
        matrix,
        rhs,
        _maxNumberOfIterations,
        _tolerance,
        &_precond,
        &solution,
        &_r,
        &_d,
        &_q,
        &_s,
        &_lastNumberOfIterations,
        &_lastResidualNorm);

    JET_INFO << "Residual after solving ICCG: " << _lastResidualNorm
             << " Number of ICCG iterations: " << _lastNumberOfIterations;

    return _lastResidualNorm <= _tolerance
        || _lastNumberOfIterations < _maxNumberOfIterations;
}

unsigned int FdmIccgSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmIccgSolver2::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmIccgSolver2::tolerance() const {
    return _tolerance;
}

double FdmIccgSolver2::lastResidual() const {
    return _lastResidualNorm;
}
