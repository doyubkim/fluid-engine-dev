// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

//
// Adopted the code from:
// http://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/
// and
// https://github.com/christopherbatty/FluidRigidCoupling2D
//

#include <pch.h>

#include <jet/constants.h>
#include <jet/fdm_iccg_solver2.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/level_set_utils.h>

using namespace jet;

const double kDefaultTolerance = 1e-6;
const double kMinWeight = 0.01;

namespace {

void restrict(const Array2<float>& finer, Array2<float>* coarser) {
    // --*--|--*--|--*--|--*--
    //  1/8   3/8   3/8   1/8
    //           to
    // -----|-----*-----|-----
    static const std::array<float, 4> centeredKernel = {
        {0.125f, 0.375f, 0.375f, 0.125f}};

    // -|----|----|----|----|-
    //      1/4  1/2  1/4
    //           to
    // -|---------|---------|-
    static const std::array<float, 4> staggeredKernel = {{0.f, 1.f, 0.f, 0.f}};

    std::array<int, 2> kernelSize;
    kernelSize[0] = finer.size().x != 2 * coarser->size().x ? 3 : 4;
    kernelSize[1] = finer.size().y != 2 * coarser->size().y ? 3 : 4;

    std::array<std::array<float, 4>, 2> kernels;
    kernels[0] = (kernelSize[0] == 3) ? staggeredKernel : centeredKernel;
    kernels[1] = (kernelSize[1] == 3) ? staggeredKernel : centeredKernel;

    const Size2 n = coarser->size();
    parallelRangeFor(
        kZeroSize, n.x, kZeroSize, n.y,
        [&](size_t iBegin, size_t iEnd, size_t jBegin, size_t jEnd) {
            std::array<size_t, 4> jIndices{{0, 0, 0, 0}};

            for (size_t j = jBegin; j < jEnd; ++j) {
                if (kernelSize[1] == 3) {
                    jIndices[0] = (j > 0) ? 2 * j - 1 : 2 * j;
                    jIndices[1] = 2 * j;
                    jIndices[2] = (j + 1 < n.y) ? 2 * j + 1 : 2 * j;
                } else {
                    jIndices[0] = (j > 0) ? 2 * j - 1 : 2 * j;
                    jIndices[1] = 2 * j;
                    jIndices[2] = 2 * j + 1;
                    jIndices[3] = (j + 1 < n.y) ? 2 * j + 2 : 2 * j + 1;
                }

                std::array<size_t, 4> iIndices{{0, 0, 0, 0}};
                for (size_t i = iBegin; i < iEnd; ++i) {
                    if (kernelSize[0] == 3) {
                        iIndices[0] = (i > 0) ? 2 * i - 1 : 2 * i;
                        iIndices[1] = 2 * i;
                        iIndices[2] = (i + 1 < n.x) ? 2 * i + 1 : 2 * i;
                    } else {
                        iIndices[0] = (i > 0) ? 2 * i - 1 : 2 * i;
                        iIndices[1] = 2 * i;
                        iIndices[2] = 2 * i + 1;
                        iIndices[3] = (i + 1 < n.x) ? 2 * i + 2 : 2 * i + 1;
                    }

                    float sum = 0.0f;
                    for (int y = 0; y < kernelSize[1]; ++y) {
                        for (int x = 0; x < kernelSize[0]; ++x) {
                            float w = kernels[0][x] * kernels[1][y];
                            sum += w * finer(iIndices[x], jIndices[y]);
                        }
                    }
                    (*coarser)(i, j) = sum;
                }
            }
        });
}

void buildSingleSystem(FdmMatrix2* A, FdmVector2* b,
                       const Array2<float>& fluidSdf,
                       const Array2<float>& uWeights,
                       const Array2<float>& vWeights,
                       std::function<Vector2D(const Vector2D&)> boundaryVel,
                       const FaceCenteredGrid2& input) {
    const Size2 size = input.resolution();
    const auto uPos = input.uPosition();
    const auto vPos = input.vPosition();

    const Vector2D invH = 1.0 / input.gridSpacing();
    const Vector2D invHSqr = invH * invH;

    // Build linear system
    A->parallelForEachIndex([&](size_t i, size_t j) {
        auto& row = (*A)(i, j);

        // initialize
        row.center = row.right = row.up = 0.0;
        (*b)(i, j) = 0.0;

        double centerPhi = fluidSdf(i, j);

        if (isInsideSdf(centerPhi)) {
            double term;

            if (i + 1 < size.x) {
                term = uWeights(i + 1, j) * invHSqr.x;
                double rightPhi = fluidSdf(i + 1, j);
                if (isInsideSdf(rightPhi)) {
                    row.center += term;
                    row.right -= term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, rightPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                (*b)(i, j) += uWeights(i + 1, j) * input.u(i + 1, j) * invH.x;
            } else {
                (*b)(i, j) += input.u(i + 1, j) * invH.x;
            }

            if (i > 0) {
                term = uWeights(i, j) * invHSqr.x;
                double leftPhi = fluidSdf(i - 1, j);
                if (isInsideSdf(leftPhi)) {
                    row.center += term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, leftPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                (*b)(i, j) -= uWeights(i, j) * input.u(i, j) * invH.x;
            } else {
                (*b)(i, j) -= input.u(i, j) * invH.x;
            }

            if (j + 1 < size.y) {
                term = vWeights(i, j + 1) * invHSqr.y;
                double upPhi = fluidSdf(i, j + 1);
                if (isInsideSdf(upPhi)) {
                    row.center += term;
                    row.up -= term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, upPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                (*b)(i, j) += vWeights(i, j + 1) * input.v(i, j + 1) * invH.y;
            } else {
                (*b)(i, j) += input.v(i, j + 1) * invH.y;
            }

            if (j > 0) {
                term = vWeights(i, j) * invHSqr.y;
                double downPhi = fluidSdf(i, j - 1);
                if (isInsideSdf(downPhi)) {
                    row.center += term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, downPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                (*b)(i, j) -= vWeights(i, j) * input.v(i, j) * invH.y;
            } else {
                (*b)(i, j) -= input.v(i, j) * invH.y;
            }

            // Accumulate contributions from the moving boundary
            double boundaryContribution =
                (1.0 - uWeights(i + 1, j)) * boundaryVel(uPos(i + 1, j)).x *
                    invH.x -
                (1.0 - uWeights(i, j)) * boundaryVel(uPos(i, j)).x * invH.x +
                (1.0 - vWeights(i, j + 1)) * boundaryVel(vPos(i, j + 1)).y *
                    invH.y -
                (1.0 - vWeights(i, j)) * boundaryVel(vPos(i, j)).y * invH.y;
            (*b)(i, j) += boundaryContribution;

            // If row.center is near-zero, the cell is likely inside a solid
            // boundary.
            if (row.center < kEpsilonD) {
                row.center = 1.0;
                (*b)(i, j) = 0.0;
            }
        } else {
            row.center = 1.0;
        }
    });
}

void buildSingleSystem(MatrixCsrD* A, VectorND* x, VectorND* b,
                       const Array2<float>& fluidSdf,
                       const Array2<float>& uWeights,
                       const Array2<float>& vWeights,
                       std::function<Vector2D(const Vector2D&)> boundaryVel,
                       const FaceCenteredGrid2& input) {
    const Size2 size = input.resolution();
    const auto uPos = input.uPosition();
    const auto vPos = input.vPosition();

    const Vector2D invH = 1.0 / input.gridSpacing();
    const Vector2D invHSqr = invH * invH;

    const auto fluidSdfAcc = fluidSdf.constAccessor();

    A->clear();
    b->clear();

    size_t numRows = 0;
    Array2<size_t> coordToIndex(size);
    fluidSdf.forEachIndex([&](size_t i, size_t j) {
        const size_t cIdx = fluidSdfAcc.index(i, j);
        const double centerPhi = fluidSdf[cIdx];

        if (isInsideSdf(centerPhi)) {
            coordToIndex[cIdx] = numRows++;
        }
    });

    fluidSdf.forEachIndex([&](size_t i, size_t j) {
        const size_t cIdx = fluidSdfAcc.index(i, j);

        const double centerPhi = fluidSdf(i, j);

        if (isInsideSdf(centerPhi)) {
            double bij = 0.0;

            std::vector<double> row(1, 0.0);
            std::vector<size_t> colIdx(1, coordToIndex[cIdx]);

            double term;

            if (i + 1 < size.x) {
                term = uWeights(i + 1, j) * invHSqr.x;
                const double rightPhi = fluidSdf(i + 1, j);
                if (isInsideSdf(rightPhi)) {
                    row[0] += term;
                    row.push_back(-term);
                    colIdx.push_back(coordToIndex(i + 1, j));
                } else {
                    double theta = fractionInsideSdf(centerPhi, rightPhi);
                    theta = std::max(theta, 0.01);
                    row[0] += term / theta;
                }
                bij += uWeights(i + 1, j) * input.u(i + 1, j) * invH.x;
            } else {
                bij += input.u(i + 1, j) * invH.x;
            }

            if (i > 0) {
                term = uWeights(i, j) * invHSqr.x;
                const double leftPhi = fluidSdf(i - 1, j);
                if (isInsideSdf(leftPhi)) {
                    row[0] += term;
                    row.push_back(-term);
                    colIdx.push_back(coordToIndex(i - 1, j));
                } else {
                    double theta = fractionInsideSdf(centerPhi, leftPhi);
                    theta = std::max(theta, 0.01);
                    row[0] += term / theta;
                }
                bij -= uWeights(i, j) * input.u(i, j) * invH.x;
            } else {
                bij -= input.u(i, j) * invH.x;
            }

            if (j + 1 < size.y) {
                term = vWeights(i, j + 1) * invHSqr.y;
                const double upPhi = fluidSdf(i, j + 1);
                if (isInsideSdf(upPhi)) {
                    row[0] += term;
                    row.push_back(-term);
                    colIdx.push_back(coordToIndex(i, j + 1));
                } else {
                    double theta = fractionInsideSdf(centerPhi, upPhi);
                    theta = std::max(theta, 0.01);
                    row[0] += term / theta;
                }
                bij += vWeights(i, j + 1) * input.v(i, j + 1) * invH.y;
            } else {
                bij += input.v(i, j + 1) * invH.y;
            }

            if (j > 0) {
                term = vWeights(i, j) * invHSqr.y;
                const double downPhi = fluidSdf(i, j - 1);
                if (isInsideSdf(downPhi)) {
                    row[0] += term;
                    row.push_back(-term);
                    colIdx.push_back(coordToIndex(i, j - 1));
                } else {
                    double theta = fractionInsideSdf(centerPhi, downPhi);
                    theta = std::max(theta, 0.01);
                    row[0] += term / theta;
                }
                bij -= vWeights(i, j) * input.v(i, j) * invH.y;
            } else {
                bij -= input.v(i, j) * invH.y;
            }

            // Accumulate contributions from the moving boundary
            const double boundaryContribution =
                (1.0 - uWeights(i + 1, j)) * boundaryVel(uPos(i + 1, j)).x *
                    invH.x -
                (1.0 - uWeights(i, j)) * boundaryVel(uPos(i, j)).x * invH.x +
                (1.0 - vWeights(i, j + 1)) * boundaryVel(vPos(i, j + 1)).y *
                    invH.y -
                (1.0 - vWeights(i, j)) * boundaryVel(vPos(i, j)).y * invH.y;
            bij += boundaryContribution;

            // If row.center is near-zero, the cell is likely inside a solid
            // boundary.
            if (row[0] < kEpsilonD) {
                row[0] = 1.0;
                bij = 0.0;
            }

            A->addRow(row, colIdx);
            b->append(bij);
        }
    });

    x->resize(b->size(), 0.0);
}

}  // namespace

GridFractionalSinglePhasePressureSolver2::
    GridFractionalSinglePhasePressureSolver2() {
    _systemSolver = std::make_shared<FdmIccgSolver2>(100, kDefaultTolerance);
}

GridFractionalSinglePhasePressureSolver2::
    ~GridFractionalSinglePhasePressureSolver2() {}

void GridFractionalSinglePhasePressureSolver2::solve(
    const FaceCenteredGrid2& input, double timeIntervalInSeconds,
    FaceCenteredGrid2* output, const ScalarField2& boundarySdf,
    const VectorField2& boundaryVelocity, const ScalarField2& fluidSdf,
    bool useCompressed) {
    UNUSED_VARIABLE(timeIntervalInSeconds);

    buildWeights(input, boundarySdf, boundaryVelocity, fluidSdf);
    buildSystem(input, useCompressed);

    if (_systemSolver != nullptr) {
        // Solve the system
        if (_mgSystemSolver == nullptr) {
            if (useCompressed) {
                _system.clear();
                _systemSolver->solveCompressed(&_compSystem);
                decompressSolution();
            } else {
                _compSystem.clear();
                _systemSolver->solve(&_system);
            }
        } else {
            _mgSystemSolver->solve(&_mgSystem);
        }

        // Apply pressure gradient
        applyPressureGradient(input, output);
    }
}

GridBoundaryConditionSolver2Ptr
GridFractionalSinglePhasePressureSolver2::suggestedBoundaryConditionSolver()
    const {
    return std::make_shared<GridFractionalBoundaryConditionSolver2>();
}

const FdmLinearSystemSolver2Ptr&
GridFractionalSinglePhasePressureSolver2::linearSystemSolver() const {
    return _systemSolver;
}

void GridFractionalSinglePhasePressureSolver2::setLinearSystemSolver(
    const FdmLinearSystemSolver2Ptr& solver) {
    _systemSolver = solver;
    _mgSystemSolver = std::dynamic_pointer_cast<FdmMgSolver2>(_systemSolver);

    if (_mgSystemSolver == nullptr) {
        // In case of non-mg system, use flat structure.
        _mgSystem.clear();
    } else {
        // In case of mg system, use multi-level structure.
        _system.clear();
        _compSystem.clear();
    }
}

const FdmVector2& GridFractionalSinglePhasePressureSolver2::pressure() const {
    if (_mgSystemSolver == nullptr) {
        return _system.x;
    } else {
        return _mgSystem.x.levels.front();
    }
}

void GridFractionalSinglePhasePressureSolver2::buildWeights(
    const FaceCenteredGrid2& input, const ScalarField2& boundarySdf,
    const VectorField2& boundaryVelocity, const ScalarField2& fluidSdf) {
    auto size = input.resolution();

    // Build levels
    size_t maxLevels = 1;
    if (_mgSystemSolver != nullptr) {
        maxLevels = _mgSystemSolver->params().maxNumberOfLevels;
    }
    FdmMgUtils2::resizeArrayWithFinest(size, maxLevels, &_fluidSdf);
    _uWeights.resize(_fluidSdf.size());
    _vWeights.resize(_fluidSdf.size());
    for (size_t l = 0; l < _fluidSdf.size(); ++l) {
        _uWeights[l].resize(_fluidSdf[l].size() + Size2(1, 0));
        _vWeights[l].resize(_fluidSdf[l].size() + Size2(0, 1));
    }

    // Build top-level grids
    auto cellPos = input.cellCenterPosition();
    auto uPos = input.uPosition();
    auto vPos = input.vPosition();
    _boundaryVel = boundaryVelocity.sampler();
    Vector2D h = input.gridSpacing();

    _fluidSdf[0].parallelForEachIndex([&](size_t i, size_t j) {
        _fluidSdf[0](i, j) = static_cast<float>(fluidSdf.sample(cellPos(i, j)));
    });

    _uWeights[0].parallelForEachIndex([&](size_t i, size_t j) {
        Vector2D pt = uPos(i, j);
        double phi0 = boundarySdf.sample(pt - Vector2D(0.5 * h.x, 0.0));
        double phi1 = boundarySdf.sample(pt + Vector2D(0.5 * h.x, 0.0));
        double frac = fractionInsideSdf(phi0, phi1);
        double weight = clamp(1.0 - frac, 0.0, 1.0);

        // Clamp non-zero weight to kMinWeight. Having nearly-zero element
        // in the matrix can be an issue.
        if (weight < kMinWeight && weight > 0.0) {
            weight = kMinWeight;
        }

        _uWeights[0](i, j) = static_cast<float>(weight);
    });

    _vWeights[0].parallelForEachIndex([&](size_t i, size_t j) {
        Vector2D pt = vPos(i, j);
        double phi0 = boundarySdf.sample(pt - Vector2D(0.0, 0.5 * h.y));
        double phi1 = boundarySdf.sample(pt + Vector2D(0.0, 0.5 * h.y));
        double frac = fractionInsideSdf(phi0, phi1);
        double weight = clamp(1.0 - frac, 0.0, 1.0);

        // Clamp non-zero weight to kMinWeight. Having nearly-zero element
        // in the matrix can be an issue.
        if (weight < kMinWeight && weight > 0.0) {
            weight = kMinWeight;
        }

        _vWeights[0](i, j) = static_cast<float>(weight);
    });

    // Build sub-levels
    for (size_t l = 1; l < _fluidSdf.size(); ++l) {
        const auto& finerFluidSdf = _fluidSdf[l - 1];
        auto& coarserFluidSdf = _fluidSdf[l];
        const auto& finerUWeight = _uWeights[l - 1];
        auto& coarserUWeight = _uWeights[l];
        const auto& finerVWeight = _vWeights[l - 1];
        auto& coarserVWeight = _vWeights[l];

        // Fluid SDF
        restrict(finerFluidSdf, &coarserFluidSdf);
        restrict(finerUWeight, &coarserUWeight);
        restrict(finerVWeight, &coarserVWeight);
    }
}

void GridFractionalSinglePhasePressureSolver2::decompressSolution() {
    const auto acc = _fluidSdf[0].constAccessor();
    _system.x.resize(acc.size());

    size_t row = 0;
    _fluidSdf[0].forEachIndex([&](size_t i, size_t j) {
        if (isInsideSdf(acc(i, j))) {
            _system.x(i, j) = _compSystem.x[row];
            ++row;
        }
    });
}

void GridFractionalSinglePhasePressureSolver2::buildSystem(
    const FaceCenteredGrid2& input, bool useCompressed) {
    Size2 size = input.resolution();
    size_t numLevels = 1;

    if (_mgSystemSolver == nullptr) {
        if (!useCompressed) {
            _system.resize(size);
        }
    } else {
        // Build levels
        size_t maxLevels = _mgSystemSolver->params().maxNumberOfLevels;
        FdmMgUtils2::resizeArrayWithFinest(size, maxLevels,
                                           &_mgSystem.A.levels);
        FdmMgUtils2::resizeArrayWithFinest(size, maxLevels,
                                           &_mgSystem.x.levels);
        FdmMgUtils2::resizeArrayWithFinest(size, maxLevels,
                                           &_mgSystem.b.levels);

        numLevels = _mgSystem.A.levels.size();
    }

    // Build top level
    const FaceCenteredGrid2* finer = &input;
    if (_mgSystemSolver == nullptr) {
        if (useCompressed) {
            buildSingleSystem(&_compSystem.A, &_compSystem.x, &_compSystem.b,
                              _fluidSdf[0], _uWeights[0], _vWeights[0],
                              _boundaryVel, *finer);
        } else {
            buildSingleSystem(&_system.A, &_system.b, _fluidSdf[0],
                              _uWeights[0], _vWeights[0], _boundaryVel, *finer);
        }
    } else {
        buildSingleSystem(&_mgSystem.A.levels.front(),
                          &_mgSystem.b.levels.front(), _fluidSdf[0],
                          _uWeights[0], _vWeights[0], _boundaryVel, *finer);
    }

    // Build sub-levels
    FaceCenteredGrid2 coarser;
    for (size_t l = 1; l < numLevels; ++l) {
        auto res = finer->resolution();
        auto h = finer->gridSpacing();
        auto o = finer->origin();
        res.x = res.x >> 1;
        res.y = res.y >> 1;
        h *= 2.0;

        // Down sample
        coarser.resize(res, h, o);
        coarser.fill(finer->sampler());

        buildSingleSystem(&_mgSystem.A.levels[l], &_mgSystem.b.levels[l],
                          _fluidSdf[l], _uWeights[l], _vWeights[l],
                          _boundaryVel, coarser);

        finer = &coarser;
    }
}

void GridFractionalSinglePhasePressureSolver2::applyPressureGradient(
    const FaceCenteredGrid2& input, FaceCenteredGrid2* output) {
    Size2 size = input.resolution();
    auto u = input.uConstAccessor();
    auto v = input.vConstAccessor();
    auto u0 = output->uAccessor();
    auto v0 = output->vAccessor();

    const auto& x = pressure();

    Vector2D invH = 1.0 / input.gridSpacing();

    x.parallelForEachIndex([&](size_t i, size_t j) {
        double centerPhi = _fluidSdf[0](i, j);

        if (i + 1 < size.x && _uWeights[0](i + 1, j) > 0.0 &&
            (isInsideSdf(centerPhi) || isInsideSdf(_fluidSdf[0](i + 1, j)))) {
            double rightPhi = _fluidSdf[0](i + 1, j);
            double theta = fractionInsideSdf(centerPhi, rightPhi);
            theta = std::max(theta, 0.01);

            u0(i + 1, j) =
                u(i + 1, j) + invH.x / theta * (x(i + 1, j) - x(i, j));
        }

        if (j + 1 < size.y && _vWeights[0](i, j + 1) > 0.0 &&
            (isInsideSdf(centerPhi) || isInsideSdf(_fluidSdf[0](i, j + 1)))) {
            double upPhi = _fluidSdf[0](i, j + 1);
            double theta = fractionInsideSdf(centerPhi, upPhi);
            theta = std::max(theta, 0.01);

            v0(i, j + 1) =
                v(i, j + 1) + invH.y / theta * (x(i, j + 1) - x(i, j));
        }
    });
}
