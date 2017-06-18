// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constants.h>
#include <jet/fdm_jacobi_solver2.h>

using namespace jet;

FdmJacobiSolver2::FdmJacobiSolver2(unsigned int maxNumberOfIterations,
                                   unsigned int residualCheckInterval,
                                   double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations),
      _lastNumberOfIterations(0),
      _residualCheckInterval(residualCheckInterval),
      _tolerance(tolerance),
      _lastResidual(kMaxD) {}

bool FdmJacobiSolver2::solve(FdmLinearSystem2* system) {
    _xTemp.resize(system->x.size());
    _residual.resize(system->x.size());

    _lastNumberOfIterations = _maxNumberOfIterations;

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        relax(system->A, system->b, &system->x, &_xTemp);

        _xTemp.swap(system->x);

        if (iter != 0 && iter % _residualCheckInterval == 0) {
            FdmBlas2::residual(system->A, system->x, system->b, &_residual);

            if (FdmBlas2::l2Norm(_residual) < _tolerance) {
                _lastNumberOfIterations = iter + 1;
                break;
            }
        }
    }

    FdmBlas2::residual(system->A, system->x, system->b, &_residual);
    _lastResidual = FdmBlas2::l2Norm(_residual);

    return _lastResidual < _tolerance;
}

unsigned int FdmJacobiSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmJacobiSolver2::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmJacobiSolver2::tolerance() const { return _tolerance; }

double FdmJacobiSolver2::lastResidual() const { return _lastResidual; }

void FdmJacobiSolver2::relax(const FdmMatrix2& A, const FdmVector2& b,
                             FdmVector2* x_, FdmVector2* xTemp_) {
    Size2 size = A.size();
    FdmVector2& x = *x_;
    FdmVector2& xTemp = *xTemp_;

    A.parallelForEachIndex([&](size_t i, size_t j) {
        double r = ((i > 0) ? A(i - 1, j).right * x(i - 1, j) : 0.0) +
                   ((i + 1 < size.x) ? A(i, j).right * x(i + 1, j) : 0.0) +
                   ((j > 0) ? A(i, j - 1).up * x(i, j - 1) : 0.0) +
                   ((j + 1 < size.y) ? A(i, j).up * x(i, j + 1) : 0.0);

        xTemp(i, j) = (b(i, j) - r) / A(i, j).center;
    });
}
