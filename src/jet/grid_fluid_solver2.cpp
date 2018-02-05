// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_utils.h>
#include <jet/constant_scalar_field2.h>
#include <jet/constants.h>
#include <jet/cubic_semi_lagrangian2.h>
#include <jet/grid_backward_euler_diffusion_solver2.h>
#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/grid_fluid_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit2.h>
#include <jet/timer.h>

#include <algorithm>

using namespace jet;

GridFluidSolver2::GridFluidSolver2()
    : GridFluidSolver2({1, 1}, {1, 1}, {0, 0}) {}

GridFluidSolver2::GridFluidSolver2(const Size2& resolution,
                                   const Vector2D& gridSpacing,
                                   const Vector2D& gridOrigin) {
    _grids = std::make_shared<GridSystemData2>();
    _grids->resize(resolution, gridSpacing, gridOrigin);

    setAdvectionSolver(std::make_shared<CubicSemiLagrangian2>());
    setDiffusionSolver(std::make_shared<GridBackwardEulerDiffusionSolver2>());
    setPressureSolver(
        std::make_shared<GridFractionalSinglePhasePressureSolver2>());
    setIsUsingFixedSubTimeSteps(false);
}

GridFluidSolver2::~GridFluidSolver2() {}

const Vector2D& GridFluidSolver2::gravity() const { return _gravity; }

void GridFluidSolver2::setGravity(const Vector2D& newGravity) {
    _gravity = newGravity;
}

double GridFluidSolver2::viscosityCoefficient() const {
    return _viscosityCoefficient;
}

void GridFluidSolver2::setViscosityCoefficient(double newValue) {
    _viscosityCoefficient = std::max(newValue, 0.0);
}

double GridFluidSolver2::cfl(double timeIntervalInSeconds) const {
    auto vel = _grids->velocity();
    double maxVel = 0.0;
    vel->forEachCellIndex([&](size_t i, size_t j) {
        Vector2D v =
            vel->valueAtCellCenter(i, j) + timeIntervalInSeconds * _gravity;
        maxVel = std::max(maxVel, v.x);
        maxVel = std::max(maxVel, v.y);
    });

    Vector2D gridSpacing = _grids->gridSpacing();
    double minGridSize = std::min(gridSpacing.x, gridSpacing.y);

    return maxVel * timeIntervalInSeconds / minGridSize;
}

double GridFluidSolver2::maxCfl() const { return _maxCfl; }

void GridFluidSolver2::setMaxCfl(double newCfl) {
    _maxCfl = std::max(newCfl, kEpsilonD);
}

bool GridFluidSolver2::useCompressedLinearSystem() const {
    return _useCompressedLinearSys;
}

void GridFluidSolver2::setUseCompressedLinearSystem(bool onoff) {
    _useCompressedLinearSys = onoff;
}

const AdvectionSolver2Ptr& GridFluidSolver2::advectionSolver() const {
    return _advectionSolver;
}

void GridFluidSolver2::setAdvectionSolver(
    const AdvectionSolver2Ptr& newSolver) {
    _advectionSolver = newSolver;
}

const GridDiffusionSolver2Ptr& GridFluidSolver2::diffusionSolver() const {
    return _diffusionSolver;
}

void GridFluidSolver2::setDiffusionSolver(
    const GridDiffusionSolver2Ptr& newSolver) {
    _diffusionSolver = newSolver;
}

const GridPressureSolver2Ptr& GridFluidSolver2::pressureSolver() const {
    return _pressureSolver;
}

void GridFluidSolver2::setPressureSolver(
    const GridPressureSolver2Ptr& newSolver) {
    _pressureSolver = newSolver;
    if (_pressureSolver != nullptr) {
        _boundaryConditionSolver =
            _pressureSolver->suggestedBoundaryConditionSolver();

        // Apply domain boundary flag
        _boundaryConditionSolver->setClosedDomainBoundaryFlag(
            _closedDomainBoundaryFlag);
    }
}

int GridFluidSolver2::closedDomainBoundaryFlag() const {
    return _closedDomainBoundaryFlag;
}

void GridFluidSolver2::setClosedDomainBoundaryFlag(int flag) {
    _closedDomainBoundaryFlag = flag;
    _boundaryConditionSolver->setClosedDomainBoundaryFlag(
        _closedDomainBoundaryFlag);
}

const GridSystemData2Ptr& GridFluidSolver2::gridSystemData() const {
    return _grids;
}

void GridFluidSolver2::resizeGrid(const Size2& newSize,
                                  const Vector2D& newGridSpacing,
                                  const Vector2D& newGridOrigin) {
    _grids->resize(newSize, newGridSpacing, newGridOrigin);
}

Size2 GridFluidSolver2::resolution() const { return _grids->resolution(); }

Vector2D GridFluidSolver2::gridSpacing() const { return _grids->gridSpacing(); }

Vector2D GridFluidSolver2::gridOrigin() const { return _grids->origin(); }

const FaceCenteredGrid2Ptr& GridFluidSolver2::velocity() const {
    return _grids->velocity();
}

const Collider2Ptr& GridFluidSolver2::collider() const { return _collider; }

void GridFluidSolver2::setCollider(const Collider2Ptr& newCollider) {
    _collider = newCollider;
}

const GridEmitter2Ptr& GridFluidSolver2::emitter() const { return _emitter; }

void GridFluidSolver2::setEmitter(const GridEmitter2Ptr& newEmitter) {
    _emitter = newEmitter;
}

void GridFluidSolver2::onInitialize() {
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

void GridFluidSolver2::onAdvanceTimeStep(double timeIntervalInSeconds) {
    // The minimum grid resolution is 1x1.
    if (_grids->resolution().x == 0 || _grids->resolution().y == 0) {
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

unsigned int GridFluidSolver2::numberOfSubTimeSteps(
    double timeIntervalInSeconds) const {
    double currentCfl = cfl(timeIntervalInSeconds);
    return static_cast<unsigned int>(
        std::max(std::ceil(currentCfl / _maxCfl), 1.0));
}

void GridFluidSolver2::onBeginAdvanceTimeStep(double timeIntervalInSeconds) {
    UNUSED_VARIABLE(timeIntervalInSeconds);
}

void GridFluidSolver2::onEndAdvanceTimeStep(double timeIntervalInSeconds) {
    UNUSED_VARIABLE(timeIntervalInSeconds);
}

void GridFluidSolver2::computeExternalForces(double timeIntervalInSeconds) {
    computeGravity(timeIntervalInSeconds);
}

void GridFluidSolver2::computeViscosity(double timeIntervalInSeconds) {
    if (_diffusionSolver != nullptr && _viscosityCoefficient > kEpsilonD) {
        auto vel = velocity();
        auto vel0 = std::dynamic_pointer_cast<FaceCenteredGrid2>(vel->clone());

        _diffusionSolver->solve(*vel0, _viscosityCoefficient,
                                timeIntervalInSeconds, vel.get(),
                                *colliderSdf(), *fluidSdf());
        applyBoundaryCondition();
    }
}

void GridFluidSolver2::computePressure(double timeIntervalInSeconds) {
    if (_pressureSolver != nullptr) {
        auto vel = velocity();
        auto vel0 = std::dynamic_pointer_cast<FaceCenteredGrid2>(vel->clone());

        _pressureSolver->solve(*vel0, timeIntervalInSeconds, vel.get(),
                               *colliderSdf(), *colliderVelocityField(),
                               *fluidSdf(), _useCompressedLinearSys);
        applyBoundaryCondition();
    }
}

void GridFluidSolver2::computeAdvection(double timeIntervalInSeconds) {
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
                std::dynamic_pointer_cast<CollocatedVectorGrid2>(grid);
            auto collocated0 =
                std::dynamic_pointer_cast<CollocatedVectorGrid2>(grid0);
            if (collocated != nullptr) {
                _advectionSolver->advect(*collocated0, *vel,
                                         timeIntervalInSeconds,
                                         collocated.get(), *colliderSdf());
                extrapolateIntoCollider(collocated.get());
                continue;
            }

            auto faceCentered =
                std::dynamic_pointer_cast<FaceCenteredGrid2>(grid);
            auto faceCentered0 =
                std::dynamic_pointer_cast<FaceCenteredGrid2>(grid0);
            if (faceCentered != nullptr && faceCentered0 != nullptr) {
                _advectionSolver->advect(*faceCentered0, *vel,
                                         timeIntervalInSeconds,
                                         faceCentered.get(), *colliderSdf());
                extrapolateIntoCollider(faceCentered.get());
                continue;
            }
        }

        // Solve velocity advection
        auto vel0 = std::dynamic_pointer_cast<FaceCenteredGrid2>(vel->clone());
        _advectionSolver->advect(*vel0, *vel0, timeIntervalInSeconds, vel.get(),
                                 *colliderSdf());
        applyBoundaryCondition();
    }
}

ScalarField2Ptr GridFluidSolver2::fluidSdf() const {
    return std::make_shared<ConstantScalarField2>(-kMaxD);
}

void GridFluidSolver2::computeGravity(double timeIntervalInSeconds) {
    if (_gravity.lengthSquared() > kEpsilonD) {
        auto vel = _grids->velocity();
        auto u = vel->uAccessor();
        auto v = vel->vAccessor();

        if (std::abs(_gravity.x) > kEpsilonD) {
            vel->forEachUIndex([&](size_t i, size_t j) {
                u(i, j) += timeIntervalInSeconds * _gravity.x;
            });
        }

        if (std::abs(_gravity.y) > kEpsilonD) {
            vel->forEachVIndex([&](size_t i, size_t j) {
                v(i, j) += timeIntervalInSeconds * _gravity.y;
            });
        }

        applyBoundaryCondition();
    }
}

void GridFluidSolver2::applyBoundaryCondition() {
    auto vel = _grids->velocity();

    if (vel != nullptr && _boundaryConditionSolver != nullptr) {
        unsigned int depth = static_cast<unsigned int>(std::ceil(_maxCfl));
        _boundaryConditionSolver->constrainVelocity(vel.get(), depth);
    }
}

void GridFluidSolver2::extrapolateIntoCollider(ScalarGrid2* grid) {
    Array2<char> marker(grid->dataSize());
    auto pos = grid->dataPosition();
    marker.parallelForEachIndex([&](size_t i, size_t j) {
        if (isInsideSdf(colliderSdf()->sample(pos(i, j)))) {
            marker(i, j) = 0;
        } else {
            marker(i, j) = 1;
        }
    });

    unsigned int depth = static_cast<unsigned int>(std::ceil(_maxCfl));
    extrapolateToRegion(grid->constDataAccessor(), marker, depth,
                        grid->dataAccessor());
}

void GridFluidSolver2::extrapolateIntoCollider(CollocatedVectorGrid2* grid) {
    Array2<char> marker(grid->dataSize());
    auto pos = grid->dataPosition();
    marker.parallelForEachIndex([&](size_t i, size_t j) {
        if (isInsideSdf(colliderSdf()->sample(pos(i, j)))) {
            marker(i, j) = 0;
        } else {
            marker(i, j) = 1;
        }
    });

    unsigned int depth = static_cast<unsigned int>(std::ceil(_maxCfl));
    extrapolateToRegion(grid->constDataAccessor(), marker, depth,
                        grid->dataAccessor());
}

void GridFluidSolver2::extrapolateIntoCollider(FaceCenteredGrid2* grid) {
    auto u = grid->uAccessor();
    auto v = grid->vAccessor();
    auto uPos = grid->uPosition();
    auto vPos = grid->vPosition();

    Array2<char> uMarker(u.size());
    Array2<char> vMarker(v.size());

    uMarker.parallelForEachIndex([&](size_t i, size_t j) {
        if (isInsideSdf(colliderSdf()->sample(uPos(i, j)))) {
            uMarker(i, j) = 0;
        } else {
            uMarker(i, j) = 1;
        }
    });

    vMarker.parallelForEachIndex([&](size_t i, size_t j) {
        if (isInsideSdf(colliderSdf()->sample(vPos(i, j)))) {
            vMarker(i, j) = 0;
        } else {
            vMarker(i, j) = 1;
        }
    });

    unsigned int depth = static_cast<unsigned int>(std::ceil(_maxCfl));
    extrapolateToRegion(grid->uConstAccessor(), uMarker, depth, u);
    extrapolateToRegion(grid->vConstAccessor(), vMarker, depth, v);
}

ScalarField2Ptr GridFluidSolver2::colliderSdf() const {
    return _boundaryConditionSolver->colliderSdf();
}

VectorField2Ptr GridFluidSolver2::colliderVelocityField() const {
    return _boundaryConditionSolver->colliderVelocityField();
}

void GridFluidSolver2::beginAdvanceTimeStep(double timeIntervalInSeconds) {
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

void GridFluidSolver2::endAdvanceTimeStep(double timeIntervalInSeconds) {
    // Invoke callback
    onEndAdvanceTimeStep(timeIntervalInSeconds);
}

void GridFluidSolver2::updateCollider(double timeIntervalInSeconds) {
    if (_collider != nullptr) {
        _collider->update(currentTimeInSeconds(), timeIntervalInSeconds);
    }
}

void GridFluidSolver2::updateEmitter(double timeIntervalInSeconds) {
    if (_emitter != nullptr) {
        _emitter->update(currentTimeInSeconds(), timeIntervalInSeconds);
    }
}

GridFluidSolver2::Builder GridFluidSolver2::builder() { return Builder(); }
