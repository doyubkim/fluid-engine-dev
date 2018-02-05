// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constants.h>
#include <jet/fdm_jacobi_solver3.h>

using namespace jet;

FdmJacobiSolver3::FdmJacobiSolver3(unsigned int maxNumberOfIterations,
                                   unsigned int residualCheckInterval,
                                   double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations),
      _lastNumberOfIterations(0),
      _residualCheckInterval(residualCheckInterval),
      _tolerance(tolerance),
      _lastResidual(kMaxD) {}

bool FdmJacobiSolver3::solve(FdmLinearSystem3* system) {
    clearCompressedVectors();

    _xTemp.resize(system->x.size());
    _residual.resize(system->x.size());

    _lastNumberOfIterations = _maxNumberOfIterations;

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        relax(system->A, system->b, &system->x, &_xTemp);

        _xTemp.swap(system->x);

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

bool FdmJacobiSolver3::solveCompressed(FdmCompressedLinearSystem3* system) {
    clearUncompressedVectors();

    _xTempComp.resize(system->x.size());
    _residualComp.resize(system->x.size());

    _lastNumberOfIterations = _maxNumberOfIterations;

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        relax(system->A, system->b, &system->x, &_xTempComp);

        _xTempComp.swap(system->x);

        if (iter != 0 && iter % _residualCheckInterval == 0) {
            FdmCompressedBlas3::residual(system->A, system->x, system->b,
                                         &_residualComp);

            if (FdmCompressedBlas3::l2Norm(_residualComp) < _tolerance) {
                _lastNumberOfIterations = iter + 1;
                break;
            }
        }
    }

    FdmCompressedBlas3::residual(system->A, system->x, system->b,
                                 &_residualComp);
    _lastResidual = FdmCompressedBlas3::l2Norm(_residualComp);

    return _lastResidual < _tolerance;
}

unsigned int FdmJacobiSolver3::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmJacobiSolver3::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmJacobiSolver3::tolerance() const { return _tolerance; }

double FdmJacobiSolver3::lastResidual() const { return _lastResidual; }

void FdmJacobiSolver3::relax(const FdmMatrix3& A, const FdmVector3& b,
                             FdmVector3* x_, FdmVector3* xTemp_) {
    Size3 size = A.size();
    FdmVector3& x = *x_;
    FdmVector3& xTemp = *xTemp_;

    A.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        double r =
            ((i > 0) ? A(i - 1, j, k).right * x(i - 1, j, k) : 0.0) +
            ((i + 1 < size.x) ? A(i, j, k).right * x(i + 1, j, k) : 0.0) +
            ((j > 0) ? A(i, j - 1, k).up * x(i, j - 1, k) : 0.0) +
            ((j + 1 < size.y) ? A(i, j, k).up * x(i, j + 1, k) : 0.0) +
            ((k > 0) ? A(i, j, k - 1).front * x(i, j, k - 1) : 0.0) +
            ((k + 1 < size.z) ? A(i, j, k).front * x(i, j, k + 1) : 0.0);

        xTemp(i, j, k) = (b(i, j, k) - r) / A(i, j, k).center;
    });
}

void FdmJacobiSolver3::relax(const MatrixCsrD& A, const VectorND& b,
                             VectorND* x_, VectorND* xTemp_) {
    const auto rp = A.rowPointersBegin();
    const auto ci = A.columnIndicesBegin();
    const auto nnz = A.nonZeroBegin();

    VectorND& x = *x_;
    VectorND& xTemp = *xTemp_;

    b.parallelForEachIndex([&](size_t i) {
        const size_t rowBegin = rp[i];
        const size_t rowEnd = rp[i + 1];

        double r = 0.0;
        double diag = 1.0;
        for (size_t jj = rowBegin; jj < rowEnd; ++jj) {
            size_t j = ci[jj];

            if (i == j) {
                diag = nnz[jj];
            } else {
                r += nnz[jj] * x[j];
            }
        }

        xTemp[i] = (b[i] - r) / diag;
    });
}

void FdmJacobiSolver3::clearUncompressedVectors() {
    _xTempComp.clear();
    _residualComp.clear();
}

void FdmJacobiSolver3::clearCompressedVectors() {
    _xTemp.clear();
    _residual.clear();
}
