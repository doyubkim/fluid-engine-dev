// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/fdm_linear_system2.h>
#include <jet/math_utils.h>
#include <jet/parallel.h>

using namespace jet;

void FdmLinearSystem2::clear() {
    A.clear();
    x.clear();
    b.clear();
}

void FdmLinearSystem2::resize(const Size2& size) {
    A.resize(size);
    x.resize(size);
    b.resize(size);
}

//

void FdmCompressedLinearSystem2::clear() {
    A.clear();
    x.clear();
    b.clear();
}

//

void FdmBlas2::set(double s, FdmVector2* result) { result->set(s); }

void FdmBlas2::set(const FdmVector2& v, FdmVector2* result) { result->set(v); }

void FdmBlas2::set(double s, FdmMatrix2* result) {
    FdmMatrixRow2 row;
    row.center = row.right = row.up = s;
    result->set(row);
}

void FdmBlas2::set(const FdmMatrix2& m, FdmMatrix2* result) { result->set(m); }

double FdmBlas2::dot(const FdmVector2& a, const FdmVector2& b) {
    Size2 size = a.size();

    JET_THROW_INVALID_ARG_IF(size != b.size());

    double result = 0.0;

    for (size_t j = 0; j < size.y; ++j) {
        for (size_t i = 0; i < size.x; ++i) {
            result += a(i, j) * b(i, j);
        }
    }

    return result;
}

void FdmBlas2::axpy(double a, const FdmVector2& x, const FdmVector2& y,
                    FdmVector2* result) {
    Size2 size = x.size();

    JET_THROW_INVALID_ARG_IF(size != y.size());
    JET_THROW_INVALID_ARG_IF(size != result->size());

    x.parallelForEachIndex(
        [&](size_t i, size_t j) { (*result)(i, j) = a * x(i, j) + y(i, j); });
}

void FdmBlas2::mvm(const FdmMatrix2& m, const FdmVector2& v,
                   FdmVector2* result) {
    Size2 size = m.size();

    JET_THROW_INVALID_ARG_IF(size != v.size());
    JET_THROW_INVALID_ARG_IF(size != result->size());

    m.parallelForEachIndex([&](size_t i, size_t j) {
        (*result)(i, j) =
            m(i, j).center * v(i, j) +
            ((i > 0) ? m(i - 1, j).right * v(i - 1, j) : 0.0) +
            ((i + 1 < size.x) ? m(i, j).right * v(i + 1, j) : 0.0) +
            ((j > 0) ? m(i, j - 1).up * v(i, j - 1) : 0.0) +
            ((j + 1 < size.y) ? m(i, j).up * v(i, j + 1) : 0.0);
    });
}

void FdmBlas2::residual(const FdmMatrix2& a, const FdmVector2& x,
                        const FdmVector2& b, FdmVector2* result) {
    Size2 size = a.size();

    JET_THROW_INVALID_ARG_IF(size != x.size());
    JET_THROW_INVALID_ARG_IF(size != b.size());
    JET_THROW_INVALID_ARG_IF(size != result->size());

    a.parallelForEachIndex([&](size_t i, size_t j) {
        (*result)(i, j) =
            b(i, j) - a(i, j).center * x(i, j) -
            ((i > 0) ? a(i - 1, j).right * x(i - 1, j) : 0.0) -
            ((i + 1 < size.x) ? a(i, j).right * x(i + 1, j) : 0.0) -
            ((j > 0) ? a(i, j - 1).up * x(i, j - 1) : 0.0) -
            ((j + 1 < size.y) ? a(i, j).up * x(i, j + 1) : 0.0);
    });
}

double FdmBlas2::l2Norm(const FdmVector2& v) { return std::sqrt(dot(v, v)); }

double FdmBlas2::lInfNorm(const FdmVector2& v) {
    Size2 size = v.size();

    double result = 0.0;

    for (size_t j = 0; j < size.y; ++j) {
        for (size_t i = 0; i < size.x; ++i) {
            result = absmax(result, v(i, j));
        }
    }

    return std::fabs(result);
}

//

void FdmCompressedBlas2::set(double s, VectorND* result) { result->set(s); }

void FdmCompressedBlas2::set(const VectorND& v, VectorND* result) {
    result->set(v);
}

void FdmCompressedBlas2::set(double s, MatrixCsrD* result) { result->set(s); }

void FdmCompressedBlas2::set(const MatrixCsrD& m, MatrixCsrD* result) {
    result->set(m);
}

double FdmCompressedBlas2::dot(const VectorND& a, const VectorND& b) {
    return a.dot(b);
}

void FdmCompressedBlas2::axpy(double a, const VectorND& x, const VectorND& y,
                              VectorND* result) {
    *result = a * x + y;
}

void FdmCompressedBlas2::mvm(const MatrixCsrD& m, const VectorND& v,
                             VectorND* result) {
    const auto rp = m.rowPointersBegin();
    const auto ci = m.columnIndicesBegin();
    const auto nnz = m.nonZeroBegin();

    v.parallelForEachIndex([&](size_t i) {
        const size_t rowBegin = rp[i];
        const size_t rowEnd = rp[i + 1];

        double sum = 0.0;

        for (size_t jj = rowBegin; jj < rowEnd; ++jj) {
            size_t j = ci[jj];
            sum += nnz[jj] * v[j];
        }

        (*result)[i] = sum;
    });
}

void FdmCompressedBlas2::residual(const MatrixCsrD& a, const VectorND& x,
                                  const VectorND& b, VectorND* result) {
    const auto rp = a.rowPointersBegin();
    const auto ci = a.columnIndicesBegin();
    const auto nnz = a.nonZeroBegin();

    x.parallelForEachIndex([&](size_t i) {
        const size_t rowBegin = rp[i];
        const size_t rowEnd = rp[i + 1];

        double sum = 0.0;

        for (size_t jj = rowBegin; jj < rowEnd; ++jj) {
            size_t j = ci[jj];
            sum += nnz[jj] * x[j];
        }

        (*result)[i] = b[i] - sum;
    });
}

double FdmCompressedBlas2::l2Norm(const VectorND& v) {
    return std::sqrt(v.dot(v));
}

double FdmCompressedBlas2::lInfNorm(const VectorND& v) {
    return std::fabs(v.absmax());
}
