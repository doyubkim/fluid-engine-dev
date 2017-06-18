// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constants.h>
#include <jet/fdm_gauss_seidel_solver3.h>

using namespace jet;

FdmGaussSeidelSolver3::FdmGaussSeidelSolver3(unsigned int maxNumberOfIterations,
                                             unsigned int residualCheckInterval,
                                             double tolerance, double sorFactor,
                                             bool useRedBlackOrdering)
    : _maxNumberOfIterations(maxNumberOfIterations),
      _lastNumberOfIterations(0),
      _residualCheckInterval(residualCheckInterval),
      _tolerance(tolerance),
      _lastResidual(kMaxD),
      _sorFactor(sorFactor),
      _useRedBlackOrdering(useRedBlackOrdering) {}

bool FdmGaussSeidelSolver3::solve(FdmLinearSystem3* system) {
    _residual.resize(system->x.size());

    _lastNumberOfIterations = _maxNumberOfIterations;

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        relax(system->A, system->b, _sorFactor, &system->x);

        if (iter != 0 && iter % _residualCheckInterval == 0) {
            FdmBlas3::residual(system->A, system->x, system->b, &_residual);

            if (FdmBlas3::l2Norm(_residual) < _tolerance) {
                _lastNumberOfIterations = iter + 1;
                break;
            }
        }
    }

    FdmBlas3::residual(system->A, system->x, system->b, &_residual);
    _lastResidual = FdmBlas3::l2Norm(_residual);

    return _lastResidual < _tolerance;
}

unsigned int FdmGaussSeidelSolver3::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmGaussSeidelSolver3::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmGaussSeidelSolver3::tolerance() const { return _tolerance; }

double FdmGaussSeidelSolver3::lastResidual() const { return _lastResidual; }

double FdmGaussSeidelSolver3::sorFactor() const { return _sorFactor; }

bool FdmGaussSeidelSolver3::useRedBlackOrdering() const {
    return _useRedBlackOrdering;
}

void FdmGaussSeidelSolver3::relax(const FdmMatrix3& A, const FdmVector3& b,
                                  double sorFactor, FdmVector3* x_) {
    Size3 size = A.size();
    FdmVector3& x = *x_;

    A.forEachIndex([&](size_t i, size_t j, size_t k) {
        double r =
            ((i > 0) ? A(i - 1, j, k).right * x(i - 1, j, k) : 0.0) +
            ((i + 1 < size.x) ? A(i, j, k).right * x(i + 1, j, k) : 0.0) +
            ((j > 0) ? A(i, j - 1, k).up * x(i, j - 1, k) : 0.0) +
            ((j + 1 < size.y) ? A(i, j, k).up * x(i, j + 1, k) : 0.0) +
            ((k > 0) ? A(i, j, k - 1).front * x(i, j, k - 1) : 0.0) +
            ((k + 1 < size.z) ? A(i, j, k).front * x(i, j, k + 1) : 0.0);

        x(i, j, k) = (1.0 - sorFactor) * x(i, j, k) +
                     sorFactor * (b(i, j, k) - r) / A(i, j, k).center;
    });
}

void FdmGaussSeidelSolver3::relaxRedBlack(const FdmMatrix3& A,
                                          const FdmVector3& b, double sorFactor,
                                          FdmVector3* x_) {
    Size3 size = A.size();
    FdmVector3& x = *x_;

    A.forEachIndex([&](size_t i, size_t j, size_t k) {
        double r =
            ((i > 0) ? A(i - 1, j, k).right * x(i - 1, j, k) : 0.0) +
            ((i + 1 < size.x) ? A(i, j, k).right * x(i + 1, j, k) : 0.0) +
            ((j > 0) ? A(i, j - 1, k).up * x(i, j - 1, k) : 0.0) +
            ((j + 1 < size.y) ? A(i, j, k).up * x(i, j + 1, k) : 0.0) +
            ((k > 0) ? A(i, j, k - 1).front * x(i, j, k - 1) : 0.0) +
            ((k + 1 < size.z) ? A(i, j, k).front * x(i, j, k + 1) : 0.0);

        x(i, j, k) = (1.0 - sorFactor) * x(i, j, k) +
                     sorFactor * (b(i, j, k) - r) / A(i, j, k).center;
    });
}
