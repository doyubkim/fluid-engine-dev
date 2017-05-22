// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/fdm_mg_linear_system2.h>

using namespace jet;

size_t FdmMgLinearSystem2::numberOfLevels() const { return A.levels.size(); }

void FdmMgLinearSystem2::resizeWithCoarsest(const Size2 &coarsestResolution,
                                            size_t numberOfLevels) {
    numberOfLevels = std::max(numberOfLevels, kOneSize);

    A.levels.resize(numberOfLevels);
    x.levels.resize(numberOfLevels);
    b.levels.resize(numberOfLevels);

    // Level 0 is the finest level, thus takes coarsestResolution ^
    // numberOfLevels.
    // Level numberOfLevels - 1 is the coarsest, taking coarsestResolution.
    Size2 res = coarsestResolution;
    for (size_t level = 0; level < numberOfLevels; ++level) {
        A[numberOfLevels - level - 1].resize(res);
        x[numberOfLevels - level - 1].resize(res);
        b[numberOfLevels - level - 1].resize(res);

        res.x = res.x << 1;
        res.y = res.y << 1;
    }
}

void FdmMgLinearSystem2::resizeWithFinest(const Size2 &finestResolution,
                                          size_t maxNumberOfLevels) {
    Size2 res = finestResolution;
    size_t i = 1;
    for (; i < maxNumberOfLevels; ++i) {
        if (res.x % 2 == 0 && res.y % 2 == 0) {
            res.x = res.x >> 1;
        } else {
            break;
        }
    }
    resizeWithCoarsest(res, i);
}

void FdmMgUtils2::restrict(const FdmVector2 &finer, FdmVector2 *coarser) {
    JET_ASSERT(finer->size().x == 2 * coarser->size().x);
    JET_ASSERT(finer->size().y == 2 * coarser->size().y);

    // --*--|--*--|--*--|--*--
    //  1/8   3/8   3/8   1/8
    //           to
    // -----|-----*-----|-----
    static const std::array<double, 4> kernel = {{0.125, 0.375, 0.375, 0.125}};

    const Size2 n = coarser->size();
    parallelRangeFor(
        kZeroSize, n.x, kZeroSize, n.y,
        [&](size_t iBegin, size_t iEnd, size_t jBegin, size_t jEnd) {
            std::array<size_t, 4> jIndices;

            for (size_t j = jBegin; j < jEnd; ++j) {
                jIndices[0] = (j > 0) ? 2 * j - 1 : 2 * j;
                jIndices[1] = 2 * j;
                jIndices[2] = 2 * j + 1;
                jIndices[3] = (j + 1 < n.y) ? 2 * j + 2 : 2 * j + 1;

                std::array<size_t, 4> iIndices;
                for (size_t i = iBegin; i < iEnd; ++i) {
                    iIndices[0] = (i > 0) ? 2 * i - 1 : 2 * i;
                    iIndices[1] = 2 * i;
                    iIndices[2] = 2 * i + 1;
                    iIndices[3] = (i + 1 < n.x) ? 2 * i + 2 : 2 * i + 1;

                    double sum = 0.0;
                    for (size_t y = 0; y < 4; ++y) {
                        for (size_t x = 0; x < 4; ++x) {
                            double w = kernel[x] * kernel[y];
                            sum += w * finer(iIndices[x], jIndices[y]);
                        }
                    }
                    (*coarser)(i, j) = sum;
                }
            }
        });
}

void FdmMgUtils2::correct(const FdmVector2 &coarser, FdmVector2 *finer) {
    JET_ASSERT(finer->size().x == 2 * coarser->size().x);
    JET_ASSERT(finer->size().y == 2 * coarser->size().y);

    // -----|-----*-----|-----
    //           to
    //  1/4   3/4   3/4   1/4
    // --*--|--*--|--*--|--*--
    static const std::array<double, 4> kernel = {{0.25, 0.75, 0.75, 0.25}};

    const Size2 n = coarser.size();
    parallelRangeFor(
        kZeroSize, n.x, kZeroSize, n.y,
        [&](size_t iBegin, size_t iEnd, size_t jBegin, size_t jEnd) {
            std::array<size_t, 4> jIndices;

            for (size_t j = jBegin; j < jEnd; ++j) {
                jIndices[0] = (j > 0) ? 2 * j - 1 : 2 * j;
                jIndices[1] = 2 * j;
                jIndices[2] = 2 * j + 1;
                jIndices[3] = (j + 1 < n.y) ? 2 * j + 2 : 2 * j + 1;

                std::array<size_t, 4> iIndices;
                for (size_t i = iBegin; i < iEnd; ++i) {
                    iIndices[0] = (i > 0) ? 2 * i - 1 : 2 * i;
                    iIndices[1] = 2 * i;
                    iIndices[2] = 2 * i + 1;
                    iIndices[3] = (i + 1 < n.x) ? 2 * i + 2 : 2 * i + 1;

                    double cij = coarser(i, j);
                    for (size_t y = 0; y < 4; ++y) {
                        for (size_t x = 0; x < 4; ++x) {
                            double w = kernel[x] * kernel[y];
                            (*finer)(iIndices[x], jIndices[y]) += w * cij;
                        }
                    }
                }
            }
        }, ExecutionPolicy::kSerial);
}
