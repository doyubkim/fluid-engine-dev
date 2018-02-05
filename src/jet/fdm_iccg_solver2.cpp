// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/cg.h>
#include <jet/fdm_iccg_solver2.h>

using namespace jet;

void FdmIccgSolver2::Preconditioner::build(const FdmMatrix2& matrix) {
    Size2 size = matrix.size();
    A = matrix.constAccessor();

    d.resize(size, 0.0);
    y.resize(size, 0.0);

    matrix.forEachIndex([&](size_t i, size_t j) {
        double denom =
            matrix(i, j).center -
            ((i > 0) ? square(matrix(i - 1, j).right) * d(i - 1, j) : 0.0) -
            ((j > 0) ? square(matrix(i, j - 1).up) * d(i, j - 1) : 0.0);

        if (std::fabs(denom) > 0.0) {
            d(i, j) = 1.0 / denom;
        } else {
            d(i, j) = 0.0;
        }
    });
}

void FdmIccgSolver2::Preconditioner::solve(const FdmVector2& b, FdmVector2* x) {
    Size2 size = b.size();
    ssize_t sx = static_cast<ssize_t>(size.x);
    ssize_t sy = static_cast<ssize_t>(size.y);

    b.forEachIndex([&](size_t i, size_t j) {
        y(i, j) = (b(i, j) - ((i > 0) ? A(i - 1, j).right * y(i - 1, j) : 0.0) -
                   ((j > 0) ? A(i, j - 1).up * y(i, j - 1) : 0.0)) *
                  d(i, j);
    });

    for (ssize_t j = sy - 1; j >= 0; --j) {
        for (ssize_t i = sx - 1; i >= 0; --i) {
            (*x)(i, j) =
                (y(i, j) -
                 ((i + 1 < sx) ? A(i, j).right * (*x)(i + 1, j) : 0.0) -
                 ((j + 1 < sy) ? A(i, j).up * (*x)(i, j + 1) : 0.0)) *
                d(i, j);
        }
    }
}

//

void FdmIccgSolver2::PreconditionerCompressed::build(const MatrixCsrD& matrix) {
    size_t size = matrix.cols();
    A = &matrix;

    d.resize(size, 0.0);
    y.resize(size, 0.0);

    const auto rp = A->rowPointersBegin();
    const auto ci = A->columnIndicesBegin();
    const auto nnz = A->nonZeroBegin();

    d.forEachIndex([&](size_t i) {
        const size_t rowBegin = rp[i];
        const size_t rowEnd = rp[i + 1];

        double denom = 0.0;
        for (size_t jj = rowBegin; jj < rowEnd; ++jj) {
            size_t j = ci[jj];

            if (j == i) {
                denom += nnz[jj];
            } else if (j < i) {
                denom -= square(nnz[jj]) * d[j];
            }
        }

        if (std::fabs(denom) > 0.0) {
            d[i] = 1.0 / denom;
        } else {
            d[i] = 0.0;
        }
    });
}

void FdmIccgSolver2::PreconditionerCompressed::solve(const VectorND& b,
                                                     VectorND* x) {
    const ssize_t size = static_cast<ssize_t>(b.size());

    const auto rp = A->rowPointersBegin();
    const auto ci = A->columnIndicesBegin();
    const auto nnz = A->nonZeroBegin();

    b.forEachIndex([&](size_t i) {
        const size_t rowBegin = rp[i];
        const size_t rowEnd = rp[i + 1];

        double sum = b[i];
        for (size_t jj = rowBegin; jj < rowEnd; ++jj) {
            size_t j = ci[jj];

            if (j < i) {
                sum -= nnz[jj] * y[j];
            }
        }

        y[i] = sum * d[i];
    });

    for (ssize_t i = size - 1; i >= 0; --i) {
        const size_t rowBegin = rp[i];
        const size_t rowEnd = rp[i + 1];

        double sum = y[i];
        for (size_t jj = rowBegin; jj < rowEnd; ++jj) {
            ssize_t j = static_cast<ssize_t>(ci[jj]);

            if (j > i) {
                sum -= nnz[jj] * (*x)[j];
            }
        }

        (*x)[i] = sum * d[i];
    }
}

//

FdmIccgSolver2::FdmIccgSolver2(unsigned int maxNumberOfIterations,
                               double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations),
      _lastNumberOfIterations(0),
      _tolerance(tolerance),
      _lastResidualNorm(kMaxD) {}

bool FdmIccgSolver2::solve(FdmLinearSystem2* system) {
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

    _precond.build(matrix);

    pcg<FdmBlas2, Preconditioner>(
        matrix, rhs, _maxNumberOfIterations, _tolerance, &_precond, &solution,
        &_r, &_d, &_q, &_s, &_lastNumberOfIterations, &_lastResidualNorm);

    JET_INFO << "Residual after solving ICCG: " << _lastResidualNorm
             << " Number of ICCG iterations: " << _lastNumberOfIterations;

    return _lastResidualNorm <= _tolerance ||
           _lastNumberOfIterations < _maxNumberOfIterations;
}

bool FdmIccgSolver2::solveCompressed(FdmCompressedLinearSystem2* system) {
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

    _precondComp.build(matrix);

    pcg<FdmCompressedBlas2, PreconditionerCompressed>(
        matrix, rhs, _maxNumberOfIterations, _tolerance, &_precondComp,
        &solution, &_rComp, &_dComp, &_qComp, &_sComp, &_lastNumberOfIterations,
        &_lastResidualNorm);

    JET_INFO << "Residual after solving ICCG: " << _lastResidualNorm
             << " Number of ICCG iterations: " << _lastNumberOfIterations;

    return _lastResidualNorm <= _tolerance ||
           _lastNumberOfIterations < _maxNumberOfIterations;
}

unsigned int FdmIccgSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmIccgSolver2::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmIccgSolver2::tolerance() const { return _tolerance; }

double FdmIccgSolver2::lastResidual() const { return _lastResidualNorm; }

void FdmIccgSolver2::clearUncompressedVectors() {
    _r.clear();
    _d.clear();
    _q.clear();
    _s.clear();
}
void FdmIccgSolver2::clearCompressedVectors() {
    _r.clear();
    _d.clear();
    _q.clear();
    _s.clear();
}