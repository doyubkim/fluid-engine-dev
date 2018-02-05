// Copyright (c) 2018 Doyub Kim
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
    clearCompressedVectors();

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

bool FdmJacobiSolver2::solveCompressed(FdmCompressedLinearSystem2* system) {
    clearUncompressedVectors();

    _xTempComp.resize(system->x.size());
    _residualComp.resize(system->x.size());

    _lastNumberOfIterations = _maxNumberOfIterations;

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        relax(system->A, system->b, &system->x, &_xTempComp);

        _xTempComp.swap(system->x);

        if (iter != 0 && iter % _residualCheckInterval == 0) {
            FdmCompressedBlas2::residual(system->A, system->x, system->b,
                                         &_residualComp);

            if (FdmCompressedBlas2::l2Norm(_residualComp) < _tolerance) {
                _lastNumberOfIterations = iter + 1;
                break;
            }
        }
    }

    FdmCompressedBlas2::residual(system->A, system->x, system->b,
                                 &_residualComp);
    _lastResidual = FdmCompressedBlas2::l2Norm(_residualComp);

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

void FdmJacobiSolver2::relax(const MatrixCsrD& A, const VectorND& b,
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

void FdmJacobiSolver2::clearUncompressedVectors() {
    _xTempComp.clear();
    _residualComp.clear();
}

void FdmJacobiSolver2::clearCompressedVectors() {
    _xTemp.clear();
    _residual.clear();
}
