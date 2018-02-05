// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cg.h>
#include <jet/constants.h>
#include <jet/fdm_iccg_solver3.h>
#include <pch.h>

using namespace jet;

void FdmIccgSolver3::Preconditioner::build(const FdmMatrix3& matrix) {
    Size3 size = matrix.size();
    A = matrix.constAccessor();

    d.resize(size, 0.0);
    y.resize(size, 0.0);

    matrix.forEachIndex([&](size_t i, size_t j, size_t k) {
        double denom =
            matrix(i, j, k).center -
            ((i > 0) ? square(matrix(i - 1, j, k).right) * d(i - 1, j, k)
                     : 0.0) -
            ((j > 0) ? square(matrix(i, j - 1, k).up) * d(i, j - 1, k) : 0.0) -
            ((k > 0) ? square(matrix(i, j, k - 1).front) * d(i, j, k - 1)
                     : 0.0);

        if (std::fabs(denom) > 0.0) {
            d(i, j, k) = 1.0 / denom;
        } else {
            d(i, j, k) = 0.0;
        }
    });
}

void FdmIccgSolver3::Preconditioner::solve(const FdmVector3& b, FdmVector3* x) {
    Size3 size = b.size();
    ssize_t sx = static_cast<ssize_t>(size.x);
    ssize_t sy = static_cast<ssize_t>(size.y);
    ssize_t sz = static_cast<ssize_t>(size.z);

    b.forEachIndex([&](size_t i, size_t j, size_t k) {
        y(i, j, k) = (b(i, j, k) -
                      ((i > 0) ? A(i - 1, j, k).right * y(i - 1, j, k) : 0.0) -
                      ((j > 0) ? A(i, j - 1, k).up * y(i, j - 1, k) : 0.0) -
                      ((k > 0) ? A(i, j, k - 1).front * y(i, j, k - 1) : 0.0)) *
                     d(i, j, k);
    });

    for (ssize_t k = sz - 1; k >= 0; --k) {
        for (ssize_t j = sy - 1; j >= 0; --j) {
            for (ssize_t i = sx - 1; i >= 0; --i) {
                (*x)(i, j, k) =
                    (y(i, j, k) -
                     ((i + 1 < sx) ? A(i, j, k).right * (*x)(i + 1, j, k)
                                   : 0.0) -
                     ((j + 1 < sy) ? A(i, j, k).up * (*x)(i, j + 1, k) : 0.0) -
                     ((k + 1 < sz) ? A(i, j, k).front * (*x)(i, j, k + 1)
                                   : 0.0)) *
                    d(i, j, k);
            }
        }
    }
}

//

void FdmIccgSolver3::PreconditionerCompressed::build(const MatrixCsrD& matrix) {
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

void FdmIccgSolver3::PreconditionerCompressed::solve(const VectorND& b,
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

FdmIccgSolver3::FdmIccgSolver3(unsigned int maxNumberOfIterations,
                               double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations),
      _lastNumberOfIterations(0),
      _tolerance(tolerance),
      _lastResidualNorm(kMaxD) {}

bool FdmIccgSolver3::solve(FdmLinearSystem3* system) {
    FdmMatrix3& matrix = system->A;
    FdmVector3& solution = system->x;
    FdmVector3& rhs = system->b;

    JET_ASSERT(matrix.size() == rhs.size());
    JET_ASSERT(matrix.size() == solution.size());

    clearCompressedVectors();

    Size3 size = matrix.size();
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

    pcg<FdmBlas3, Preconditioner>(
        matrix, rhs, _maxNumberOfIterations, _tolerance, &_precond, &solution,
        &_r, &_d, &_q, &_s, &_lastNumberOfIterations, &_lastResidualNorm);

    JET_INFO << "Residual norm after solving ICCG: " << _lastResidualNorm
             << " Number of ICCG iterations: " << _lastNumberOfIterations;

    return _lastResidualNorm <= _tolerance ||
           _lastNumberOfIterations < _maxNumberOfIterations;
}

bool FdmIccgSolver3::solveCompressed(FdmCompressedLinearSystem3* system) {
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

    pcg<FdmCompressedBlas3, PreconditionerCompressed>(
        matrix, rhs, _maxNumberOfIterations, _tolerance, &_precondComp,
        &solution, &_rComp, &_dComp, &_qComp, &_sComp, &_lastNumberOfIterations,
        &_lastResidualNorm);

    JET_INFO << "Residual after solving ICCG: " << _lastResidualNorm
             << " Number of ICCG iterations: " << _lastNumberOfIterations;

    return _lastResidualNorm <= _tolerance ||
           _lastNumberOfIterations < _maxNumberOfIterations;
}

unsigned int FdmIccgSolver3::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

unsigned int FdmIccgSolver3::lastNumberOfIterations() const {
    return _lastNumberOfIterations;
}

double FdmIccgSolver3::tolerance() const { return _tolerance; }

double FdmIccgSolver3::lastResidual() const { return _lastResidualNorm; }

void FdmIccgSolver3::clearUncompressedVectors() {
    _r.clear();
    _d.clear();
    _q.clear();
    _s.clear();
}
void FdmIccgSolver3::clearCompressedVectors() {
    _r.clear();
    _d.clear();
    _q.clear();
    _s.clear();
}