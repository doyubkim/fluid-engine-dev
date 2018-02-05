// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/grid_smoke_solver2.h>

#include <algorithm>

using namespace jet;

GridSmokeSolver2::GridSmokeSolver2()
    : GridSmokeSolver2({1, 1}, {1, 1}, {0, 0}) {}

GridSmokeSolver2::GridSmokeSolver2(const Size2& resolution,
                                   const Vector2D& gridSpacing,
                                   const Vector2D& gridOrigin)
    : GridFluidSolver2(resolution, gridSpacing, gridOrigin) {
    auto grids = gridSystemData();
    _smokeDensityDataId = grids->addAdvectableScalarData(
        std::make_shared<CellCenteredScalarGrid2::Builder>(), 0.0);
    _temperatureDataId = grids->addAdvectableScalarData(
        std::make_shared<CellCenteredScalarGrid2::Builder>(), 0.0);
}

GridSmokeSolver2::~GridSmokeSolver2() {}

double GridSmokeSolver2::smokeDiffusionCoefficient() const {
    return _smokeDiffusionCoefficient;
}

void GridSmokeSolver2::setSmokeDiffusionCoefficient(double newValue) {
    _smokeDiffusionCoefficient = std::max(newValue, 0.0);
}

double GridSmokeSolver2::temperatureDiffusionCoefficient() const {
    return _temperatureDiffusionCoefficient;
}

void GridSmokeSolver2::setTemperatureDiffusionCoefficient(double newValue) {
    _temperatureDiffusionCoefficient = std::max(newValue, 0.0);
}

double GridSmokeSolver2::buoyancySmokeDensityFactor() const {
    return _buoyancySmokeDensityFactor;
}

void GridSmokeSolver2::setBuoyancySmokeDensityFactor(double newValue) {
    _buoyancySmokeDensityFactor = newValue;
}

double GridSmokeSolver2::buoyancyTemperatureFactor() const {
    return _buoyancyTemperatureFactor;
}

void GridSmokeSolver2::setBuoyancyTemperatureFactor(double newValue) {
    _buoyancyTemperatureFactor = newValue;
}

double GridSmokeSolver2::smokeDecayFactor() const { return _smokeDecayFactor; }

void GridSmokeSolver2::setSmokeDecayFactor(double newValue) {
    _smokeDecayFactor = clamp(newValue, 0.0, 1.0);
}

double GridSmokeSolver2::smokeTemperatureDecayFactor() const {
    return _temperatureDecayFactor;
}

void GridSmokeSolver2::setTemperatureDecayFactor(double newValue) {
    _temperatureDecayFactor = clamp(newValue, 0.0, 1.0);
}

ScalarGrid2Ptr GridSmokeSolver2::smokeDensity() const {
    return gridSystemData()->advectableScalarDataAt(_smokeDensityDataId);
}

ScalarGrid2Ptr GridSmokeSolver2::temperature() const {
    return gridSystemData()->advectableScalarDataAt(_temperatureDataId);
}

void GridSmokeSolver2::onEndAdvanceTimeStep(double timeIntervalInSeconds) {
    computeDiffusion(timeIntervalInSeconds);
}

void GridSmokeSolver2::computeExternalForces(double timeIntervalInSeconds) {
    computeBuoyancyForce(timeIntervalInSeconds);
}

void GridSmokeSolver2::computeDiffusion(double timeIntervalInSeconds) {
    if (diffusionSolver() != nullptr) {
        if (_smokeDiffusionCoefficient > kEpsilonD) {
            auto den = smokeDensity();
            auto den0 = std::dynamic_pointer_cast<CellCenteredScalarGrid2>(
                den->clone());

            diffusionSolver()->solve(*den0, _smokeDiffusionCoefficient,
                                     timeIntervalInSeconds, den.get(),
                                     *colliderSdf());
            extrapolateIntoCollider(den.get());
        }

        if (_temperatureDiffusionCoefficient > kEpsilonD) {
            auto temp = smokeDensity();
            auto temp0 = std::dynamic_pointer_cast<CellCenteredScalarGrid2>(
                temp->clone());

            diffusionSolver()->solve(*temp0, _temperatureDiffusionCoefficient,
                                     timeIntervalInSeconds, temp.get(),
                                     *colliderSdf());
            extrapolateIntoCollider(temp.get());
        }
    }

    auto den = smokeDensity();
    den->parallelForEachDataPointIndex(
        [&](size_t i, size_t j) { (*den)(i, j) *= 1.0 - _smokeDecayFactor; });
    auto temp = temperature();
    temp->parallelForEachDataPointIndex([&](size_t i, size_t j) {
        (*temp)(i, j) *= 1.0 - _temperatureDecayFactor;
    });
}

void GridSmokeSolver2::computeBuoyancyForce(double timeIntervalInSeconds) {
    auto grids = gridSystemData();
    auto vel = grids->velocity();

    Vector2D up(0, 1);
    if (gravity().lengthSquared() > kEpsilonD) {
        up = -gravity().normalized();
    }

    if (std::abs(_buoyancySmokeDensityFactor) > kEpsilonD ||
        std::abs(_buoyancyTemperatureFactor) > kEpsilonD) {
        auto den = smokeDensity();
        auto temp = temperature();

        double tAmb = 0.0;
        temp->forEachCellIndex(
            [&](size_t i, size_t j) { tAmb += (*temp)(i, j); });
        tAmb /=
            static_cast<double>(temp->resolution().x * temp->resolution().y);

        auto u = vel->uAccessor();
        auto v = vel->vAccessor();
        auto uPos = vel->uPosition();
        auto vPos = vel->vPosition();

        if (std::abs(up.x) > kEpsilonD) {
            vel->parallelForEachUIndex([&](size_t i, size_t j) {
                Vector2D pt = uPos(i, j);
                double fBuoy =
                    _buoyancySmokeDensityFactor * den->sample(pt) +
                    _buoyancyTemperatureFactor * (temp->sample(pt) - tAmb);
                u(i, j) += timeIntervalInSeconds * fBuoy * up.x;
            });
        }

        if (std::abs(up.y) > kEpsilonD) {
            vel->parallelForEachVIndex([&](size_t i, size_t j) {
                Vector2D pt = vPos(i, j);
                double fBuoy =
                    _buoyancySmokeDensityFactor * den->sample(pt) +
                    _buoyancyTemperatureFactor * (temp->sample(pt) - tAmb);
                v(i, j) += timeIntervalInSeconds * fBuoy * up.y;
            });
        }

        applyBoundaryCondition();
    }
}

GridSmokeSolver2::Builder GridSmokeSolver2::builder() { return Builder(); }

GridSmokeSolver2 GridSmokeSolver2::Builder::build() const {
    return GridSmokeSolver2(_resolution, getGridSpacing(), _gridOrigin);
}

GridSmokeSolver2Ptr GridSmokeSolver2::Builder::makeShared() const {
    return std::shared_ptr<GridSmokeSolver2>(
        new GridSmokeSolver2(_resolution, getGridSpacing(), _gridOrigin),
        [](GridSmokeSolver2* obj) { delete obj; });
}
