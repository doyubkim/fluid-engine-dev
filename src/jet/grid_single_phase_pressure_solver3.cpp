// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constants.h>
#include <jet/fdm_iccg_solver3.h>
#include <jet/grid_blocked_boundary_condition_solver3.h>
#include <jet/grid_single_phase_pressure_solver3.h>
#include <jet/level_set_utils.h>

using namespace jet;

const char kFluid = 0;
const char kAir = 1;
const char kBoundary = 2;

const double kDefaultTolerance = 1e-6;

namespace {

void buildSingleSystem(FdmMatrix3* A, FdmVector3* b,
                       const Array3<char>& markers,
                       const FaceCenteredGrid3& input) {
    Size3 size = input.resolution();
    Vector3D invH = 1.0 / input.gridSpacing();
    Vector3D invHSqr = invH * invH;

    // Build linear system
    A->parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        auto& row = (*A)(i, j, k);

        // initialize
        row.center = row.right = row.up = row.front = 0.0;
        (*b)(i, j, k) = 0.0;

        if (markers(i, j, k) == kFluid) {
            (*b)(i, j, k) = input.divergenceAtCellCenter(i, j, k);

            if (i + 1 < size.x && markers(i + 1, j, k) != kBoundary) {
                row.center += invHSqr.x;
                if (markers(i + 1, j, k) == kFluid) {
                    row.right -= invHSqr.x;
                }
            }

            if (i > 0 && markers(i - 1, j, k) != kBoundary) {
                row.center += invHSqr.x;
            }

            if (j + 1 < size.y && markers(i, j + 1, k) != kBoundary) {
                row.center += invHSqr.y;
                if (markers(i, j + 1, k) == kFluid) {
                    row.up -= invHSqr.y;
                }
            }

            if (j > 0 && markers(i, j - 1, k) != kBoundary) {
                row.center += invHSqr.y;
            }

            if (k + 1 < size.z && markers(i, j, k + 1) != kBoundary) {
                row.center += invHSqr.z;
                if (markers(i, j, k + 1) == kFluid) {
                    row.front -= invHSqr.z;
                }
            }

            if (k > 0 && markers(i, j, k - 1) != kBoundary) {
                row.center += invHSqr.z;
            }
        } else {
            row.center = 1.0;
        }
    });
}

void buildSingleSystem(MatrixCsrD* A, VectorND* x, VectorND* b,
                       const Array3<char>& markers,
                       const FaceCenteredGrid3& input) {
    Size3 size = input.resolution();
    Vector3D invH = 1.0 / input.gridSpacing();
    Vector3D invHSqr = invH * invH;

    const auto markerAcc = markers.constAccessor();

    A->clear();
    b->clear();

    size_t numRows = 0;
    Array3<size_t> coordToIndex(size);
    markers.forEachIndex([&](size_t i, size_t j, size_t k) {
        const size_t cIdx = markerAcc.index(i, j, k);

        if (markerAcc[cIdx] == kFluid) {
            coordToIndex[cIdx] = numRows++;
        }
    });

    markers.forEachIndex([&](size_t i, size_t j, size_t k) {
        const size_t cIdx = markerAcc.index(i, j, k);

        if (markerAcc[cIdx] == kFluid) {
            b->append(input.divergenceAtCellCenter(i, j, k));

            std::vector<double> row(1, 0.0);
            std::vector<size_t> colIdx(1, coordToIndex[cIdx]);

            if (i + 1 < size.x && markers(i + 1, j, k) != kBoundary) {
                row[0] += invHSqr.x;
                const size_t rIdx = markerAcc.index(i + 1, j, k);
                if (markers[rIdx] == kFluid) {
                    row.push_back(-invHSqr.x);
                    colIdx.push_back(coordToIndex[rIdx]);
                }
            }

            if (i > 0 && markers(i - 1, j, k) != kBoundary) {
                row[0] += invHSqr.x;
                const size_t lIdx = markerAcc.index(i - 1, j, k);
                if (markers[lIdx] == kFluid) {
                    row.push_back(-invHSqr.x);
                    colIdx.push_back(coordToIndex[lIdx]);
                }
            }

            if (j + 1 < size.y && markers(i, j + 1, k) != kBoundary) {
                row[0] += invHSqr.y;
                const size_t uIdx = markerAcc.index(i, j + 1, k);
                if (markers[uIdx] == kFluid) {
                    row.push_back(-invHSqr.y);
                    colIdx.push_back(coordToIndex[uIdx]);
                }
            }

            if (j > 0 && markers(i, j - 1, k) != kBoundary) {
                row[0] += invHSqr.y;
                const size_t dIdx = markerAcc.index(i, j - 1, k);
                if (markers[dIdx] == kFluid) {
                    row.push_back(-invHSqr.y);
                    colIdx.push_back(coordToIndex[dIdx]);
                }
            }

            if (k + 1 < size.z && markers(i, j, k + 1) != kBoundary) {
                row[0] += invHSqr.z;
                const size_t fIdx = markerAcc.index(i, j, k + 1);
                if (markers[fIdx] == kFluid) {
                    row.push_back(-invHSqr.z);
                    colIdx.push_back(coordToIndex[fIdx]);
                }
            }

            if (k > 0 && markers(i, j, k - 1) != kBoundary) {
                row[0] += invHSqr.z;
                const size_t bIdx = markerAcc.index(i, j, k - 1);
                if (markers[bIdx] == kFluid) {
                    row.push_back(-invHSqr.z);
                    colIdx.push_back(coordToIndex[bIdx]);
                }
            }

            A->addRow(row, colIdx);
        }
    });

    x->resize(b->size(), 0.0);
}

}  // namespace

GridSinglePhasePressureSolver3::GridSinglePhasePressureSolver3() {
    _systemSolver = std::make_shared<FdmIccgSolver3>(100, kDefaultTolerance);
}

GridSinglePhasePressureSolver3::~GridSinglePhasePressureSolver3() {}

void GridSinglePhasePressureSolver3::solve(const FaceCenteredGrid3& input,
                                           double timeIntervalInSeconds,
                                           FaceCenteredGrid3* output,
                                           const ScalarField3& boundarySdf,
                                           const VectorField3& boundaryVelocity,
                                           const ScalarField3& fluidSdf,
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

GridBoundaryConditionSolver3Ptr
GridSinglePhasePressureSolver3::suggestedBoundaryConditionSolver() const {
    return std::make_shared<GridBlockedBoundaryConditionSolver3>();
}

const FdmLinearSystemSolver3Ptr&
GridSinglePhasePressureSolver3::linearSystemSolver() const {
    return _systemSolver;
}

void GridSinglePhasePressureSolver3::setLinearSystemSolver(
    const FdmLinearSystemSolver3Ptr& solver) {
    _systemSolver = solver;
    _mgSystemSolver = std::dynamic_pointer_cast<FdmMgSolver3>(_systemSolver);

    if (_mgSystemSolver == nullptr) {
        // In case of non-mg system, use flat structure.
        _mgSystem.clear();
    } else {
        // In case of mg system, use multi-level structure.
        _system.clear();
        _compSystem.clear();
    }
}

const FdmVector3& GridSinglePhasePressureSolver3::pressure() const {
    if (_mgSystemSolver == nullptr) {
        return _system.x;
    } else {
        return _mgSystem.x.levels.front();
    }
}

void GridSinglePhasePressureSolver3::buildMarkers(
    const Size3& size,
    const std::function<Vector3D(size_t, size_t, size_t)>& pos,
    const ScalarField3& boundarySdf, const ScalarField3& fluidSdf) {
    // Build levels
    size_t maxLevels = 1;
    if (_mgSystemSolver != nullptr) {
        maxLevels = _mgSystemSolver->params().maxNumberOfLevels;
    }
    FdmMgUtils3::resizeArrayWithFinest(size, maxLevels, &_markers);

    // Build top-level markers
    _markers[0].parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = pos(i, j, k);
        if (isInsideSdf(boundarySdf.sample(pt))) {
            _markers[0](i, j, k) = kBoundary;
        } else if (isInsideSdf(fluidSdf.sample(pt))) {
            _markers[0](i, j, k) = kFluid;
        } else {
            _markers[0](i, j, k) = kAir;
        }
    });

    // Build sub-level markers
    for (size_t l = 1; l < _markers.size(); ++l) {
        const auto& finer = _markers[l - 1];
        auto& coarser = _markers[l];
        const Size3 n = coarser.size();

        parallelRangeFor(
            kZeroSize, n.x, kZeroSize, n.y, kZeroSize, n.z,
            [&](size_t iBegin, size_t iEnd, size_t jBegin, size_t jEnd,
                size_t kBegin, size_t kEnd) {
                std::array<size_t, 4> kIndices;

                for (size_t k = kBegin; k < kEnd; ++k) {
                    kIndices[0] = (k > 0) ? 2 * k - 1 : 2 * k;
                    kIndices[1] = 2 * k;
                    kIndices[2] = 2 * k + 1;
                    kIndices[3] = (k + 1 < n.z) ? 2 * k + 2 : 2 * k + 1;

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
                            for (size_t z = 0; z < 4; ++z) {
                                for (size_t y = 0; y < 4; ++y) {
                                    for (size_t x = 0; x < 4; ++x) {
                                        char f = finer(iIndices[x], jIndices[y],
                                                       kIndices[z]);
                                        if (f == kBoundary) {
                                            ++cnt[(int)kBoundary];
                                        } else if (f == kFluid) {
                                            ++cnt[(int)kFluid];
                                        } else {
                                            ++cnt[(int)kAir];
                                        }
                                    }
                                }
                            }

                            coarser(i, j, k) = static_cast<char>(
                                argmax3(cnt[0], cnt[1], cnt[2]));
                        }
                    }
                }
            });
    }
}

void GridSinglePhasePressureSolver3::decompressSolution() {
    const auto acc = _markers[0].constAccessor();
    _system.x.resize(acc.size());

    size_t row = 0;
    _markers[0].forEachIndex([&](size_t i, size_t j, size_t k) {
        if (acc(i, j, k) == kFluid) {
            _system.x(i, j, k) = _compSystem.x[row];
            ++row;
        }
    });
}

void GridSinglePhasePressureSolver3::buildSystem(const FaceCenteredGrid3& input,
                                                 bool useCompressed) {
    Size3 size = input.resolution();
    size_t numLevels = 1;

    if (_mgSystemSolver == nullptr) {
        if (!useCompressed) {
            _system.resize(size);
        }
    } else {
        // Build levels
        size_t maxLevels = _mgSystemSolver->params().maxNumberOfLevels;
        FdmMgUtils3::resizeArrayWithFinest(size, maxLevels,
                                           &_mgSystem.A.levels);
        FdmMgUtils3::resizeArrayWithFinest(size, maxLevels,
                                           &_mgSystem.x.levels);
        FdmMgUtils3::resizeArrayWithFinest(size, maxLevels,
                                           &_mgSystem.b.levels);

        numLevels = _mgSystem.A.levels.size();
    }

    // Build top level
    const FaceCenteredGrid3* finer = &input;
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
    FaceCenteredGrid3 coarser;
    for (size_t l = 1; l < numLevels; ++l) {
        auto res = finer->resolution();
        auto h = finer->gridSpacing();
        auto o = finer->origin();
        res.x = res.x >> 1;
        res.y = res.y >> 1;
        res.z = res.z >> 1;
        h *= 2.0;

        // Down sample
        coarser.resize(res, h, o);
        coarser.fill(finer->sampler());

        buildSingleSystem(&_mgSystem.A.levels[l], &_mgSystem.b.levels[l],
                          _markers[l], coarser);

        finer = &coarser;
    }
}

void GridSinglePhasePressureSolver3::applyPressureGradient(
    const FaceCenteredGrid3& input, FaceCenteredGrid3* output) {
    Size3 size = input.resolution();
    auto u = input.uConstAccessor();
    auto v = input.vConstAccessor();
    auto w = input.wConstAccessor();
    auto u0 = output->uAccessor();
    auto v0 = output->vAccessor();
    auto w0 = output->wAccessor();

    const auto& x = pressure();

    Vector3D invH = 1.0 / input.gridSpacing();

    x.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (_markers[0](i, j, k) == kFluid) {
            if (i + 1 < size.x && _markers[0](i + 1, j, k) != kBoundary) {
                u0(i + 1, j, k) =
                    u(i + 1, j, k) + invH.x * (x(i + 1, j, k) - x(i, j, k));
            }
            if (j + 1 < size.y && _markers[0](i, j + 1, k) != kBoundary) {
                v0(i, j + 1, k) =
                    v(i, j + 1, k) + invH.y * (x(i, j + 1, k) - x(i, j, k));
            }
            if (k + 1 < size.z && _markers[0](i, j, k + 1) != kBoundary) {
                w0(i, j, k + 1) =
                    w(i, j, k + 1) + invH.z * (x(i, j, k + 1) - x(i, j, k));
            }
        }
    });
}
