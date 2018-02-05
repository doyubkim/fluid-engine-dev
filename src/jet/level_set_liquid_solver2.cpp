// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/array_utils.h>
#include <jet/eno_level_set_solver2.h>
#include <jet/fmm_level_set_solver2.h>
#include <jet/level_set_liquid_solver2.h>
#include <jet/level_set_utils.h>
#include <jet/timer.h>

#include <algorithm>

using namespace jet;

LevelSetLiquidSolver2::LevelSetLiquidSolver2()
: LevelSetLiquidSolver2({1, 1}, {1, 1}, {0, 0}) {
}

LevelSetLiquidSolver2::LevelSetLiquidSolver2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin)
: GridFluidSolver2(resolution, gridSpacing, gridOrigin) {
    auto grids = gridSystemData();
    _signedDistanceFieldId = grids->addAdvectableScalarData(
        std::make_shared<CellCenteredScalarGrid2::Builder>(), kMaxD);
    _levelSetSolver = std::make_shared<EnoLevelSetSolver2>();
}

LevelSetLiquidSolver2::~LevelSetLiquidSolver2() {
}

ScalarGrid2Ptr LevelSetLiquidSolver2::signedDistanceField() const {
    return gridSystemData()->advectableScalarDataAt(_signedDistanceFieldId);
}

LevelSetSolver2Ptr LevelSetLiquidSolver2::levelSetSolver() const {
    return _levelSetSolver;
}

void LevelSetLiquidSolver2::setLevelSetSolver(
    const LevelSetSolver2Ptr& newSolver) {
    _levelSetSolver = newSolver;
}

void LevelSetLiquidSolver2::setMinReinitializeDistance(double distance) {
    _minReinitializeDistance = distance;
}

void LevelSetLiquidSolver2::setIsGlobalCompensationEnabled(bool isEnabled) {
    _isGlobalCompensationEnabled = isEnabled;
}

double LevelSetLiquidSolver2::computeVolume() const {
    auto sdf = signedDistanceField();
    const Vector2D gridSpacing = sdf->gridSpacing();
    const double cellVolume = gridSpacing.x * gridSpacing.y;
    const double h = std::max(gridSpacing.x, gridSpacing.y);

    double volume = 0.0;
    sdf->forEachDataPointIndex([&](size_t i, size_t j) {
        volume += 1.0 - smearedHeavisideSdf((*sdf)(i, j) / h);
    });
    volume *= cellVolume;

    return volume;
}

void LevelSetLiquidSolver2::onBeginAdvanceTimeStep(
    double timeIntervalInSeconds) {
    UNUSED_VARIABLE(timeIntervalInSeconds);

    // Measure current volume
    _lastKnownVolume = computeVolume();

    JET_INFO << "Current volume: " << _lastKnownVolume;
}

void LevelSetLiquidSolver2::onEndAdvanceTimeStep(double timeIntervalInSeconds) {
    double currentCfl = cfl(timeIntervalInSeconds);

    Timer timer;
    reinitialize(currentCfl);
    JET_INFO << "reinitializing level set field took "
             << timer.durationInSeconds() << " seconds";

    // Measure current volume
    double currentVol = computeVolume();
    double volDiff = currentVol - _lastKnownVolume;

    JET_INFO << "Current volume: " << currentVol << " "
             << "Volume diff: " << volDiff;

    if (_isGlobalCompensationEnabled) {
        addVolume(-volDiff);

        currentVol = computeVolume();
        JET_INFO << "Volume after global compensation: " << currentVol;
    }
}

void LevelSetLiquidSolver2::computeAdvection(double timeIntervalInSeconds) {
    double currentCfl = cfl(timeIntervalInSeconds);

    Timer timer;
    extrapolateVelocityToAir(currentCfl);
    JET_INFO << "velocity extrapolation took "
             << timer.durationInSeconds() << " seconds";

    GridFluidSolver2::computeAdvection(timeIntervalInSeconds);
}

ScalarField2Ptr LevelSetLiquidSolver2::fluidSdf() const {
    return signedDistanceField();
}

void LevelSetLiquidSolver2::reinitialize(double currentCfl) {
    if (_levelSetSolver != nullptr) {
        auto sdf = signedDistanceField();
        auto sdf0 = sdf->clone();

        const Vector2D gridSpacing = sdf->gridSpacing();
        const double h = std::max(gridSpacing.x, gridSpacing.y);
        const double maxReinitDist
            = std::max(2.0 * currentCfl, _minReinitializeDistance) * h;

        JET_INFO << "Max reinitialize distance: " << maxReinitDist;

        _levelSetSolver->reinitialize(
            *sdf0, maxReinitDist, sdf.get());
        extrapolateIntoCollider(sdf.get());
    }
}

void LevelSetLiquidSolver2::extrapolateVelocityToAir(double currentCfl) {
    auto sdf = signedDistanceField();
    auto vel = gridSystemData()->velocity();

    auto u = vel->uAccessor();
    auto v = vel->vAccessor();
    auto uPos = vel->uPosition();
    auto vPos = vel->vPosition();

    Array2<char> uMarker(u.size());
    Array2<char> vMarker(v.size());

    uMarker.parallelForEachIndex([&](size_t i, size_t j) {
        if (isInsideSdf(sdf->sample(uPos(i, j)))) {
            uMarker(i, j) = 1;
        } else {
            uMarker(i, j) = 0;
            u(i, j) = 0.0;
        }
    });

    vMarker.parallelForEachIndex([&](size_t i, size_t j) {
        if (isInsideSdf(sdf->sample(vPos(i, j)))) {
            vMarker(i, j) = 1;
        } else {
            vMarker(i, j) = 0;
            v(i, j) = 0.0;
        }
    });

    const Vector2D gridSpacing = sdf->gridSpacing();
    const double h = std::max(gridSpacing.x, gridSpacing.y);
    const double maxDist
        = std::max(2.0 * currentCfl, _minReinitializeDistance) * h;

    JET_INFO << "Max velocity extrapolation distance: " << maxDist;

    FmmLevelSetSolver2 fmmSolver;
    fmmSolver.extrapolate(*vel, *sdf, maxDist, vel.get());

    applyBoundaryCondition();
}

void LevelSetLiquidSolver2::addVolume(double volDiff) {
    auto sdf = signedDistanceField();
    const Vector2D gridSpacing = sdf->gridSpacing();
    const double cellVolume = gridSpacing.x * gridSpacing.y;
    const double h = std::max(gridSpacing.x, gridSpacing.y);

    double volume0 = 0.0;
    double volume1 = 0.0;
    sdf->forEachDataPointIndex([&](size_t i, size_t j) {
        volume0 += 1.0 - smearedHeavisideSdf((*sdf)(i, j) / h);
        volume1 += 1.0 - smearedHeavisideSdf((*sdf)(i, j) / h + 1.0);
    });
    volume0 *= cellVolume;
    volume1 *= cellVolume;

    const double dVdh = (volume1 - volume0) / h;

    if (std::abs(dVdh) > 0.0) {
        double dist = volDiff / dVdh;

        sdf->parallelForEachDataPointIndex([&](size_t i, size_t j) {
            (*sdf)(i, j) += dist;
        });
    }
}

LevelSetLiquidSolver2::Builder LevelSetLiquidSolver2::builder() {
    return Builder();
}


LevelSetLiquidSolver2 LevelSetLiquidSolver2::Builder::build() const {
    return LevelSetLiquidSolver2(
        _resolution,
        getGridSpacing(),
        _gridOrigin);
}

LevelSetLiquidSolver2Ptr LevelSetLiquidSolver2::Builder::makeShared() const {
    return std::shared_ptr<LevelSetLiquidSolver2>(
        new LevelSetLiquidSolver2(
            _resolution,
            getGridSpacing(),
            _gridOrigin),
        [] (LevelSetLiquidSolver2* obj) {
            delete obj;
        });
}
