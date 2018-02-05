// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constants.h>
#include <jet/fdm_iccg_solver2.h>
#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/grid_single_phase_pressure_solver2.h>
#include <jet/level_set_utils.h>

using namespace jet;

const char kFluid = 0;
const char kAir = 1;
const char kBoundary = 2;

const double kDefaultTolerance = 1e-6;

namespace {

void buildSingleSystem(FdmMatrix2* A, FdmVector2* b,
                       const Array2<char>& markers,
                       const FaceCenteredGrid2& input) {
    Size2 size = input.resolution();
    Vector2D invH = 1.0 / input.gridSpacing();
    Vector2D invHSqr = invH * invH;

    A->parallelForEachIndex([&](size_t i, size_t j) {
        auto& row = (*A)(i, j);

        // initialize
        row.center = row.right = row.up = 0.0;
        (*b)(i, j) = 0.0;

        if (markers(i, j) == kFluid) {
            (*b)(i, j) = input.divergenceAtCellCenter(i, j);

            if (i + 1 < size.x && markers(i + 1, j) != kBoundary) {
                row.center += invHSqr.x;
                if (markers(i + 1, j) == kFluid) {
                    row.right -= invHSqr.x;
                }
            }

            if (i > 0 && markers(i - 1, j) != kBoundary) {
                row.center += invHSqr.x;
            }

            if (j + 1 < size.y && markers(i, j + 1) != kBoundary) {
                row.center += invHSqr.y;
                if (markers(i, j + 1) == kFluid) {
                    row.up -= invHSqr.y;
                }
            }

            if (j > 0 && markers(i, j - 1) != kBoundary) {
                row.center += invHSqr.y;
            }
        } else {
            row.center = 1.0;
        }
    });
}

void buildSingleSystem(MatrixCsrD* A, VectorND* x, VectorND* b,
                       const Array2<char>& markers,
                       const FaceCenteredGrid2& input) {
    Size2 size = input.resolution();
    Vector2D invH = 1.0 / input.gridSpacing();
    Vector2D invHSqr = invH * invH;

    const auto markerAcc = markers.constAccessor();

    A->clear();
    b->clear();

    size_t numRows = 0;
    Array2<size_t> coordToIndex(size);
    markers.forEachIndex([&](size_t i, size_t j) {
        const size_t cIdx = markerAcc.index(i, j);

        if (markerAcc[cIdx] == kFluid) {
            coordToIndex[cIdx] = numRows++;
        }
    });

    markers.forEachIndex([&](size_t i, size_t j) {
        const size_t cIdx = markerAcc.index(i, j);

        if (markerAcc[cIdx] == kFluid) {
            b->append(input.divergenceAtCellCenter(i, j));

            std::vector<double> row(1, 0.0);
            std::vector<size_t> colIdx(1, coordToIndex[cIdx]);

            if (i + 1 < size.x && markers(i + 1, j) != kBoundary) {
                row[0] += invHSqr.x;
                const size_t rIdx = markerAcc.index(i + 1, j);
                if (markers[rIdx] == kFluid) {
                    row.push_back(-invHSqr.x);
                    colIdx.push_back(coordToIndex[rIdx]);
                }
            }

            if (i > 0 && markers(i - 1, j) != kBoundary) {
                row[0] += invHSqr.x;
                const size_t lIdx = markerAcc.index(i - 1, j);
                if (markers[lIdx] == kFluid) {
                    row.push_back(-invHSqr.x);
                    colIdx.push_back(coordToIndex[lIdx]);
                }
            }

            if (j + 1 < size.y && markers(i, j + 1) != kBoundary) {
                row[0] += invHSqr.y;
                const size_t uIdx = markerAcc.index(i, j + 1);
                if (markers[uIdx] == kFluid) {
                    row.push_back(-invHSqr.y);
                    colIdx.push_back(coordToIndex[uIdx]);
                }
            }

            if (j > 0 && markers(i, j - 1) != kBoundary) {
                row[0] += invHSqr.y;
                const size_t dIdx = markerAcc.index(i, j - 1);
                if (markers[dIdx] == kFluid) {
                    row.push_back(-invHSqr.y);
                    colIdx.push_back(coordToIndex[dIdx]);
                }
            }

            A->addRow(row, colIdx);
        }
    });

    x->resize(b->size(), 0.0);
}

}  // namespace

GridSinglePhasePressureSolver2::GridSinglePhasePressureSolver2() {
    _systemSolver = std::make_shared<FdmIccgSolver2>(100, kDefaultTolerance);
}

GridSinglePhasePressureSolver2::~GridSinglePhasePressureSolver2() {}

void GridSinglePhasePressureSolver2::solve(const FaceCenteredGrid2& input,
                                           double timeIntervalInSeconds,
                                           FaceCenteredGrid2* output,
                                           const ScalarField2& boundarySdf,
                                           const VectorField2& boundaryVelocity,
                                           const ScalarField2& fluidSdf,
                                           bool useCompressed) {
    UNUSED_VARIABLE(timeIntervalInSeconds);
    UNUSED_VARIABLE(boundaryVelocity);

    auto pos = input.cellCenterPosition();
    buildMarkers(input.resolution(), pos, boundarySdf, fluidSdf);
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
GridSinglePhasePressureSolver2::suggestedBoundaryConditionSolver() const {
    return std::make_shared<GridBlockedBoundaryConditionSolver2>();
}

const FdmLinearSystemSolver2Ptr&
GridSinglePhasePressureSolver2::linearSystemSolver() const {
    return _systemSolver;
}

void GridSinglePhasePressureSolver2::setLinearSystemSolver(
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

const FdmVector2& GridSinglePhasePressureSolver2::pressure() const {
    if (_mgSystemSolver == nullptr) {
        return _system.x;
    } else {
        return _mgSystem.x.levels.front();
    }
}

void GridSinglePhasePressureSolver2::buildMarkers(
    const Size2& size, const std::function<Vector2D(size_t, size_t)>& pos,
    const ScalarField2& boundarySdf, const ScalarField2& fluidSdf) {
    // Build levels
    size_t maxLevels = 1;
    if (_mgSystemSolver != nullptr) {
        maxLevels = _mgSystemSolver->params().maxNumberOfLevels;
    }
    FdmMgUtils2::resizeArrayWithFinest(size, maxLevels, &_markers);

    // Build top-level markers
    _markers[0].parallelForEachIndex([&](size_t i, size_t j) {
        Vector2D pt = pos(i, j);
        if (isInsideSdf(boundarySdf.sample(pt))) {
            _markers[0](i, j) = kBoundary;
        } else if (isInsideSdf(fluidSdf.sample(pt))) {
            _markers[0](i, j) = kFluid;
        } else {
            _markers[0](i, j) = kAir;
        }
    });

    // Build sub-level markers
    for (size_t l = 1; l < _markers.size(); ++l) {
        const auto& finer = _markers[l - 1];
        auto& coarser = _markers[l];
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

                        int cnt[3] = {0, 0, 0};
                        for (size_t y = 0; y < 4; ++y) {
                            for (size_t x = 0; x < 4; ++x) {
                                char f = finer(iIndices[x], jIndices[y]);
                                if (f == kBoundary) {
                                    ++cnt[(int)kBoundary];
                                } else if (f == kFluid) {
                                    ++cnt[(int)kFluid];
                                } else {
                                    ++cnt[(int)kAir];
                                }
                            }
                        }

                        coarser(i, j) =
                            static_cast<char>(argmax3(cnt[0], cnt[1], cnt[2]));
                    }
                }
            });
    }
}

void GridSinglePhasePressureSolver2::decompressSolution() {
    const auto acc = _markers[0].constAccessor();
    _system.x.resize(acc.size());

    size_t row = 0;
    _markers[0].forEachIndex([&](size_t i, size_t j) {
        if (acc(i, j) == kFluid) {
            _system.x(i, j) = _compSystem.x[row];
            ++row;
        }
    });
}

void GridSinglePhasePressureSolver2::buildSystem(const FaceCenteredGrid2& input,
                                                 bool useCompressed) {
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
                              _markers[0], *finer);
        } else {
            buildSingleSystem(&_system.A, &_system.b, _markers[0], *finer);
        }
    } else {
        buildSingleSystem(&_mgSystem.A.levels.front(),
                          &_mgSystem.b.levels.front(), _markers[0], *finer);
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
                          _markers[l], coarser);

        finer = &coarser;
    }
}

void GridSinglePhasePressureSolver2::applyPressureGradient(
    const FaceCenteredGrid2& input, FaceCenteredGrid2* output) {
    Size2 size = input.resolution();
    auto u = input.uConstAccessor();
    auto v = input.vConstAccessor();
    auto u0 = output->uAccessor();
    auto v0 = output->vAccessor();

    const auto& x = pressure();

    Vector2D invH = 1.0 / input.gridSpacing();

    x.parallelForEachIndex([&](size_t i, size_t j) {
        if (_markers[0](i, j) == kFluid) {
            if (i + 1 < size.x && _markers[0](i + 1, j) != kBoundary) {
                u0(i + 1, j) = u(i + 1, j) + invH.x * (x(i + 1, j) - x(i, j));
            }
            if (j + 1 < size.y && _markers[0](i, j + 1) != kBoundary) {
                v0(i, j + 1) = v(i, j + 1) + invH.y * (x(i, j + 1) - x(i, j));
            }
        }
    });
}
