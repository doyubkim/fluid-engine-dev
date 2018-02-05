// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constants.h>
#include <jet/fdm_gauss_seidel_solver2.h>

using namespace jet;

FdmGaussSeidelSolver2::FdmGaussSeidelSolver2(unsigned int maxNumberOfIterations,
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

bool FdmGaussSeidelSolver2::solve(FdmLinearSystem2* system) {
    clearCompressedVectors();

    _residual.resize(system->x.size());

    _lastNumberOfIterations = _maxNumberOfIterations;

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        if (_useRedBlackOrdering) {
            relaxRedBlack(system->A, system->b, _sorFactor, &system->x);
        } else {
            relax(system->A, system->b, _sorFactor, &system->x);
        }

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

bool FdmGaussSeidelSolver2::solveCompressed(
    FdmCompressedLinearSystem2* system) {
    clearUncompressedVectors();

    _residualComp.resize(system->x.size());

    _lastNumberOfIterations = _maxNumberOfIterations;

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        relax(system->A, system->b, _sorFactor, &system->x);

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

unsigned int FdmGaussSeidelSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmGaussSeidelSolver2::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmGaussSeidelSolver2::tolerance() const { return _tolerance; }

double FdmGaussSeidelSolver2::lastResidual() const { return _lastResidual; }

double FdmGaussSeidelSolver2::sorFactor() const { return _sorFactor; }

bool FdmGaussSeidelSolver2::useRedBlackOrdering() const {
    return _useRedBlackOrdering;
}

void FdmGaussSeidelSolver2::relax(const FdmMatrix2& A, const FdmVector2& b,
                                  double sorFactor, FdmVector2* x_) {
    Size2 size = A.size();
    FdmVector2& x = *x_;

    A.forEachIndex([&](size_t i, size_t j) {
        double r = ((i > 0) ? A(i - 1, j).right * x(i - 1, j) : 0.0) +
                   ((i + 1 < size.x) ? A(i, j).right * x(i + 1, j) : 0.0) +
                   ((j > 0) ? A(i, j - 1).up * x(i, j - 1) : 0.0) +
                   ((j + 1 < size.y) ? A(i, j).up * x(i, j + 1) : 0.0);

        x(i, j) = (1.0 - sorFactor) * x(i, j) +
                  sorFactor * (b(i, j) - r) / A(i, j).center;
    });
}

void FdmGaussSeidelSolver2::relax(const MatrixCsrD& A, const VectorND& b,
                                  double sorFactor, VectorND* x_) {
    const auto rp = A.rowPointersBegin();
    const auto ci = A.columnIndicesBegin();
    const auto nnz = A.nonZeroBegin();

    VectorND& x = *x_;

    b.forEachIndex([&](size_t i) {
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

        x[i] = (1.0 - sorFactor) * x[i] + sorFactor * (b[i] - r) / diag;
    });
}

void FdmGaussSeidelSolver2::relaxRedBlack(const FdmMatrix2& A,
                                          const FdmVector2& b, double sorFactor,
                                          FdmVector2* x_) {
    Size2 size = A.size();
    FdmVector2& x = *x_;

    // Red update
    parallelRangeFor(
        kZeroSize, size.x, kZeroSize, size.y,
        [&](size_t iBegin, size_t iEnd, size_t jBegin, size_t jEnd) {
            for (size_t j = jBegin; j < jEnd; ++j) {
                size_t i = j % 2 + iBegin;  // i.e. (0, 0)
                for (; i < iEnd; i += 2) {
                    double r =
                        ((i > 0) ? A(i - 1, j).right * x(i - 1, j) : 0.0) +
                        ((i + 1 < size.x) ? A(i, j).right * x(i + 1, j) : 0.0) +
                        ((j > 0) ? A(i, j - 1).up * x(i, j - 1) : 0.0) +
                        ((j + 1 < size.y) ? A(i, j).up * x(i, j + 1) : 0.0);

                    x(i, j) = (1.0 - sorFactor) * x(i, j) +
                              sorFactor * (b(i, j) - r) / A(i, j).center;
                }
            }
        });

    // Black update
    parallelRangeFor(
        kZeroSize, size.x, kZeroSize, size.y,
        [&](size_t iBegin, size_t iEnd, size_t jBegin, size_t jEnd) {
            for (size_t j = jBegin; j < jEnd; ++j) {
                size_t i = 1 - j % 2 + iBegin;  // i.e. (1, 0)
                for (; i < iEnd; i += 2) {
                    double r =
                        ((i > 0) ? A(i - 1, j).right * x(i - 1, j) : 0.0) +
                        ((i + 1 < size.x) ? A(i, j).right * x(i + 1, j) : 0.0) +
                        ((j > 0) ? A(i, j - 1).up * x(i, j - 1) : 0.0) +
                        ((j + 1 < size.y) ? A(i, j).up * x(i, j + 1) : 0.0);

                    x(i, j) = (1.0 - sorFactor) * x(i, j) +
                              sorFactor * (b(i, j) - r) / A(i, j).center;
                }
            }
        });
}

void FdmGaussSeidelSolver2::clearUncompressedVectors() { _residual.clear(); }

void FdmGaussSeidelSolver2::clearCompressedVectors() { _residualComp.clear(); }
