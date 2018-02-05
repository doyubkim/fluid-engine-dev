// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_utils.h>
#include <jet/constant_scalar_field3.h>
#include <jet/constants.h>
#include <jet/cubic_semi_lagrangian3.h>
#include <jet/grid_backward_euler_diffusion_solver3.h>
#include <jet/grid_blocked_boundary_condition_solver3.h>
#include <jet/grid_fluid_solver3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit3.h>
#include <jet/timer.h>

#include <algorithm>

using namespace jet;

GridFluidSolver3::GridFluidSolver3()
    : GridFluidSolver3({1, 1, 1}, {1, 1, 1}, {0, 0, 0}) {}

GridFluidSolver3::GridFluidSolver3(const Size3& resolution,
                                   const Vector3D& gridSpacing,
                                   const Vector3D& gridOrigin) {
    _grids = std::make_shared<GridSystemData3>();
    _grids->resize(resolution, gridSpacing, gridOrigin);

    setAdvectionSolver(std::make_shared<CubicSemiLagrangian3>());
    setDiffusionSolver(std::make_shared<GridBackwardEulerDiffusionSolver3>());
    setPressureSolver(
        std::make_shared<GridFractionalSinglePhasePressureSolver3>());
    setIsUsingFixedSubTimeSteps(false);
}

GridFluidSolver3::~GridFluidSolver3() {}

const Vector3D& GridFluidSolver3::gravity() const { return _gravity; }

void GridFluidSolver3::setGravity(const Vector3D& newGravity) {
    _gravity = newGravity;
}

double GridFluidSolver3::viscosityCoefficient() const {
    return _viscosityCoefficient;
}

void GridFluidSolver3::setViscosityCoefficient(double newValue) {
    _viscosityCoefficient = std::max(newValue, 0.0);
}

double GridFluidSolver3::cfl(double timeIntervalInSeconds) const {
    auto vel = _grids->velocity();
    double maxVel = 0.0;
    vel->forEachCellIndex([&](size_t i, size_t j, size_t k) {
        Vector3D v =
            vel->valueAtCellCenter(i, j, k) + timeIntervalInSeconds * _gravity;
        maxVel = std::max(maxVel, v.x);
        maxVel = std::max(maxVel, v.y);
        maxVel = std::max(maxVel, v.z);
    });

    Vector3D gridSpacing = _grids->gridSpacing();
    double minGridSize = min3(gridSpacing.x, gridSpacing.y, gridSpacing.z);

    return maxVel * timeIntervalInSeconds / minGridSize;
}

double GridFluidSolver3::maxCfl() const { return _maxCfl; }

void GridFluidSolver3::setMaxCfl(double newCfl) {
    _maxCfl = std::max(newCfl, kEpsilonD);
}

bool GridFluidSolver3::useCompressedLinearSystem() const {
    return _useCompressedLinearSys;
}

void GridFluidSolver3::setUseCompressedLinearSystem(bool onoff) {
    _useCompressedLinearSys = onoff;
}

const AdvectionSolver3Ptr& GridFluidSolver3::advectionSolver() const {
    return _advectionSolver;
}

void GridFluidSolver3::setAdvectionSolver(
    const AdvectionSolver3Ptr& newSolver) {
    _advectionSolver = newSolver;
}

const GridDiffusionSolver3Ptr& GridFluidSolver3::diffusionSolver() const {
    return _diffusionSolver;
}

void GridFluidSolver3::setDiffusionSolver(
    const GridDiffusionSolver3Ptr& newSolver) {
    _diffusionSolver = newSolver;
}

const GridPressureSolver3Ptr& GridFluidSolver3::pressureSolver() const {
    return _pressureSolver;
}

void GridFluidSolver3::setPressureSolver(
    const GridPressureSolver3Ptr& newSolver) {
    _pressureSolver = newSolver;
    if (_pressureSolver != nullptr) {
        _boundaryConditionSolver =
            _pressureSolver->suggestedBoundaryConditionSolver();

        // Apply domain boundary flag
        _boundaryConditionSolver->setClosedDomainBoundaryFlag(
            _closedDomainBoundaryFlag);
    }
}

int GridFluidSolver3::closedDomainBoundaryFlag() const {
    return _closedDomainBoundaryFlag;
}

void GridFluidSolver3::setClosedDomainBoundaryFlag(int flag) {
    _closedDomainBoundaryFlag = flag;
    _boundaryConditionSolver->setClosedDomainBoundaryFlag(
        _closedDomainBoundaryFlag);
}

const GridSystemData3Ptr& GridFluidSolver3::gridSystemData() const {
    return _grids;
}

void GridFluidSolver3::resizeGrid(const Size3& newSize,
                                  const Vector3D& newGridSpacing,
                                  const Vector3D& newGridOrigin) {
    _grids->resize(newSize, newGridSpacing, newGridOrigin);
}

Size3 GridFluidSolver3::resolution() const { return _grids->resolution(); }

Vector3D GridFluidSolver3::gridSpacing() const { return _grids->gridSpacing(); }

Vector3D GridFluidSolver3::gridOrigin() const { return _grids->origin(); }

const FaceCenteredGrid3Ptr& GridFluidSolver3::velocity() const {
    return _grids->velocity();
}

const Collider3Ptr& GridFluidSolver3::collider() const { return _collider; }

void GridFluidSolver3::setCollider(const Collider3Ptr& newCollider) {
    _collider = newCollider;
}

const GridEmitter3Ptr& GridFluidSolver3::emitter() const { return _emitter; }

void GridFluidSolver3::setEmitter(const GridEmitter3Ptr& newEmitter) {
    _emitter = newEmitter;
}

void GridFluidSolver3::onInitialize() {
    // When initializing the solver, update the collider and emitter state as
    // well since they also affects the initial condition of the simulation.
    Timer timer;
    updateCollider(0.0);
    JET_INFO << "Update collider took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    updateEmitter(0.0);
    JET_INFO << "Update emitter took " << timer.durationInSeconds()
             << " seconds";
}

void GridFluidSolver3::onAdvanceTimeStep(double timeIntervalInSeconds) {
    // The minimum grid resolution is 1x1.
    if (_grids->resolution().x == 0 || _grids->resolution().y == 0 ||
        _grids->resolution().z == 0) {
        JET_WARN << "Empty grid. Skipping the simulation.";
        return;
    }

    beginAdvanceTimeStep(timeIntervalInSeconds);

    Timer timer;
    computeExternalForces(timeIntervalInSeconds);
    JET_INFO << "Computing external force took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    computeViscosity(timeIntervalInSeconds);
    JET_INFO << "Computing viscosity force took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    computePressure(timeIntervalInSeconds);
    JET_INFO << "Computing pressure force took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    computeAdvection(timeIntervalInSeconds);
    JET_INFO << "Computing advection force took " << timer.durationInSeconds()
             << " seconds";

    endAdvanceTimeStep(timeIntervalInSeconds);
}

unsigned int GridFluidSolver3::numberOfSubTimeSteps(
    double timeIntervalInSeconds) const {
    double currentCfl = cfl(timeIntervalInSeconds);
    return static_cast<unsigned int>(
        std::max(std::ceil(currentCfl / _maxCfl), 1.0));
}

void GridFluidSolver3::onBeginAdvanceTimeStep(double timeIntervalInSeconds) {
    UNUSED_VARIABLE(timeIntervalInSeconds);
}

void GridFluidSolver3::onEndAdvanceTimeStep(double timeIntervalInSeconds) {
    UNUSED_VARIABLE(timeIntervalInSeconds);
}

void GridFluidSolver3::computeExternalForces(double timeIntervalInSeconds) {
    computeGravity(timeIntervalInSeconds);
}

void GridFluidSolver3::computeViscosity(double timeIntervalInSeconds) {
    if (_diffusionSolver != nullptr && _viscosityCoefficient > kEpsilonD) {
        auto vel = velocity();
        auto vel0 = std::dynamic_pointer_cast<FaceCenteredGrid3>(vel->clone());

        _diffusionSolver->solve(*vel0, _viscosityCoefficient,
                                timeIntervalInSeconds, vel.get(),
                                *colliderSdf(), *fluidSdf());
        applyBoundaryCondition();
    }
}

void GridFluidSolver3::computePressure(double timeIntervalInSeconds) {
    if (_pressureSolver != nullptr) {
        auto vel = velocity();
        auto vel0 = std::dynamic_pointer_cast<FaceCenteredGrid3>(vel->clone());

        _pressureSolver->solve(*vel0, timeIntervalInSeconds, vel.get(),
                               *colliderSdf(), *colliderVelocityField(),
                               *fluidSdf(), _useCompressedLinearSys);
        applyBoundaryCondition();
    }
}

void GridFluidSolver3::computeAdvection(double timeIntervalInSeconds) {
    auto vel = velocity();
    if (_advectionSolver != nullptr) {
        // Solve advections for custom scalar fields
        size_t n = _grids->numberOfAdvectableScalarData();
        for (size_t i = 0; i < n; ++i) {
            auto grid = _grids->advectableScalarDataAt(i);
            auto grid0 = grid->clone();
            _advectionSolver->advect(*grid0, *vel, timeIntervalInSeconds,
                                     grid.get(), *colliderSdf());
            extrapolateIntoCollider(grid.get());
        }

        // Solve advections for custom vector fields
        n = _grids->numberOfAdvectableVectorData();
        size_t velIdx = _grids->velocityIndex();
        for (size_t i = 0; i < n; ++i) {
            // Handle velocity layer separately.
            if (i == velIdx) {
                continue;
            }

            auto grid = _grids->advectableVectorDataAt(i);
            auto grid0 = grid->clone();

            auto collocated =
                std::dynamic_pointer_cast<CollocatedVectorGrid3>(grid);
            auto collocated0 =
                std::dynamic_pointer_cast<CollocatedVectorGrid3>(grid0);
            if (collocated != nullptr) {
                _advectionSolver->advect(*collocated0, *vel,
                                         timeIntervalInSeconds,
                                         collocated.get(), *colliderSdf());
                extrapolateIntoCollider(collocated.get());
                continue;
            }

            auto faceCentered =
                std::dynamic_pointer_cast<FaceCenteredGrid3>(grid);
            auto faceCentered0 =
                std::dynamic_pointer_cast<FaceCenteredGrid3>(grid0);
            if (faceCentered != nullptr && faceCentered0 != nullptr) {
                _advectionSolver->advect(*faceCentered0, *vel,
                                         timeIntervalInSeconds,
                                         faceCentered.get(), *colliderSdf());
                extrapolateIntoCollider(faceCentered.get());
                continue;
            }
        }

        // Solve velocity advection
        auto vel0 = std::dynamic_pointer_cast<FaceCenteredGrid3>(vel->clone());
        _advectionSolver->advect(*vel0, *vel0, timeIntervalInSeconds, vel.get(),
                                 *colliderSdf());
        applyBoundaryCondition();
    }
}

ScalarField3Ptr GridFluidSolver3::fluidSdf() const {
    return std::make_shared<ConstantScalarField3>(-kMaxD);
}

void GridFluidSolver3::computeGravity(double timeIntervalInSeconds) {
    if (_gravity.lengthSquared() > kEpsilonD) {
        auto vel = _grids->velocity();
        auto u = vel->uAccessor();
        auto v = vel->vAccessor();
        auto w = vel->wAccessor();

        if (std::abs(_gravity.x) > kEpsilonD) {
            vel->forEachUIndex([&](size_t i, size_t j, size_t k) {
                u(i, j, k) += timeIntervalInSeconds * _gravity.x;
            });
        }

        if (std::abs(_gravity.y) > kEpsilonD) {
            vel->forEachVIndex([&](size_t i, size_t j, size_t k) {
                v(i, j, k) += timeIntervalInSeconds * _gravity.y;
            });
        }

        if (std::abs(_gravity.z) > kEpsilonD) {
            vel->forEachWIndex([&](size_t i, size_t j, size_t k) {
                w(i, j, k) += timeIntervalInSeconds * _gravity.z;
            });
        }

        applyBoundaryCondition();
    }
}

void GridFluidSolver3::applyBoundaryCondition() {
    auto vel = _grids->velocity();

    if (vel != nullptr && _boundaryConditionSolver != nullptr) {
        unsigned int depth = static_cast<unsigned int>(std::ceil(_maxCfl));
        _boundaryConditionSolver->constrainVelocity(vel.get(), depth);
    }
}

void GridFluidSolver3::extrapolateIntoCollider(ScalarGrid3* grid) {
    Array3<char> marker(grid->dataSize());
    auto pos = grid->dataPosition();
    marker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(colliderSdf()->sample(pos(i, j, k)))) {
            marker(i, j, k) = 0;
        } else {
            marker(i, j, k) = 1;
        }
    });

    unsigned int depth = static_cast<unsigned int>(std::ceil(_maxCfl));
    extrapolateToRegion(grid->constDataAccessor(), marker, depth,
                        grid->dataAccessor());
}

void GridFluidSolver3::extrapolateIntoCollider(CollocatedVectorGrid3* grid) {
    Array3<char> marker(grid->dataSize());
    auto pos = grid->dataPosition();
    marker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(colliderSdf()->sample(pos(i, j, k)))) {
            marker(i, j, k) = 0;
        } else {
            marker(i, j, k) = 1;
        }
    });

    unsigned int depth = static_cast<unsigned int>(std::ceil(_maxCfl));
    extrapolateToRegion(grid->constDataAccessor(), marker, depth,
                        grid->dataAccessor());
}

void GridFluidSolver3::extrapolateIntoCollider(FaceCenteredGrid3* grid) {
    auto u = grid->uAccessor();
    auto v = grid->vAccessor();
    auto w = grid->wAccessor();
    auto uPos = grid->uPosition();
    auto vPos = grid->vPosition();
    auto wPos = grid->wPosition();

    Array3<char> uMarker(u.size());
    Array3<char> vMarker(v.size());
    Array3<char> wMarker(w.size());

    uMarker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(colliderSdf()->sample(uPos(i, j, k)))) {
            uMarker(i, j, k) = 0;
        } else {
            uMarker(i, j, k) = 1;
        }
    });

    vMarker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(colliderSdf()->sample(vPos(i, j, k)))) {
            vMarker(i, j, k) = 0;
        } else {
            vMarker(i, j, k) = 1;
        }
    });

    wMarker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(colliderSdf()->sample(wPos(i, j, k)))) {
            wMarker(i, j, k) = 0;
        } else {
            wMarker(i, j, k) = 1;
        }
    });

    unsigned int depth = static_cast<unsigned int>(std::ceil(_maxCfl));
    extrapolateToRegion(grid->uConstAccessor(), uMarker, depth, u);
    extrapolateToRegion(grid->vConstAccessor(), vMarker, depth, v);
    extrapolateToRegion(grid->wConstAccessor(), wMarker, depth, w);
}

ScalarField3Ptr GridFluidSolver3::colliderSdf() const {
    return _boundaryConditionSolver->colliderSdf();
}

VectorField3Ptr GridFluidSolver3::colliderVelocityField() const {
    return _boundaryConditionSolver->colliderVelocityField();
}

void GridFluidSolver3::beginAdvanceTimeStep(double timeIntervalInSeconds) {
    // Update collider and emitter
    Timer timer;
    updateCollider(timeIntervalInSeconds);
    JET_INFO << "Update collider took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    updateEmitter(timeIntervalInSeconds);
    JET_INFO << "Update emitter took " << timer.durationInSeconds()
             << " seconds";

    // Update boundary condition solver
    if (_boundaryConditionSolver != nullptr) {
        _boundaryConditionSolver->updateCollider(
            _collider, _grids->resolution(), _grids->gridSpacing(),
            _grids->origin());
    }

    // Apply boundary condition to the velocity field in case the field got
    // updated externally.
    applyBoundaryCondition();

    // Invoke callback
    onBeginAdvanceTimeStep(timeIntervalInSeconds);
}

void GridFluidSolver3::endAdvanceTimeStep(double timeIntervalInSeconds) {
    // Invoke callback
    onEndAdvanceTimeStep(timeIntervalInSeconds);
}

void GridFluidSolver3::updateCollider(double timeIntervalInSeconds) {
    if (_collider != nullptr) {
        _collider->update(currentTimeInSeconds(), timeIntervalInSeconds);
    }
}

void GridFluidSolver3::updateEmitter(double timeIntervalInSeconds) {
    if (_emitter != nullptr) {
        _emitter->update(currentTimeInSeconds(), timeIntervalInSeconds);
    }
}

GridFluidSolver3::Builder GridFluidSolver3::builder() { return Builder(); }
