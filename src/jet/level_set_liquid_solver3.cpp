// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/array_utils.h>
#include <jet/eno_level_set_solver3.h>
#include <jet/fmm_level_set_solver3.h>
#include <jet/level_set_liquid_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/timer.h>

#include <algorithm>

using namespace jet;

LevelSetLiquidSolver3::LevelSetLiquidSolver3()
: LevelSetLiquidSolver3({1, 1, 1}, {1, 1, 1}, {0, 0, 0}) {
}

LevelSetLiquidSolver3::LevelSetLiquidSolver3(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& gridOrigin)
: GridFluidSolver3(resolution, gridSpacing, gridOrigin) {
    auto grids = gridSystemData();
    _signedDistanceFieldId = grids->addAdvectableScalarData(
        std::make_shared<CellCenteredScalarGrid3::Builder>(), kMaxD);
    _levelSetSolver = std::make_shared<EnoLevelSetSolver3>();
}

LevelSetLiquidSolver3::~LevelSetLiquidSolver3() {
}

ScalarGrid3Ptr LevelSetLiquidSolver3::signedDistanceField() const {
    return gridSystemData()->advectableScalarDataAt(_signedDistanceFieldId);
}

LevelSetSolver3Ptr LevelSetLiquidSolver3::levelSetSolver() const {
    return _levelSetSolver;
}

void LevelSetLiquidSolver3::setLevelSetSolver(
    const LevelSetSolver3Ptr& newSolver) {
    _levelSetSolver = newSolver;
}

void LevelSetLiquidSolver3::setMinReinitializeDistance(double distance) {
    _minReinitializeDistance = distance;
}

void LevelSetLiquidSolver3::setIsGlobalCompensationEnabled(bool isEnabled) {
    _isGlobalCompensationEnabled = isEnabled;
}

double LevelSetLiquidSolver3::computeVolume() const {
    auto sdf = signedDistanceField();
    const Vector3D gridSpacing = sdf->gridSpacing();
    const double cellVolume = gridSpacing.x * gridSpacing.y * gridSpacing.z;
    const double h = max3(gridSpacing.x, gridSpacing.y, gridSpacing.z);

    double volume = 0.0;
    sdf->forEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        volume += 1.0 - smearedHeavisideSdf((*sdf)(i, j, k) / h);
    });
    volume *= cellVolume;

    return volume;
}

void LevelSetLiquidSolver3::onBeginAdvanceTimeStep(
    double timeIntervalInSeconds) {
    UNUSED_VARIABLE(timeIntervalInSeconds);

    // Measure current volume
    _lastKnownVolume = computeVolume();

    JET_INFO << "Current volume: " << _lastKnownVolume;
}

void LevelSetLiquidSolver3::onEndAdvanceTimeStep(double timeIntervalInSeconds) {
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

void LevelSetLiquidSolver3::computeAdvection(double timeIntervalInSeconds) {
    double currentCfl = cfl(timeIntervalInSeconds);

    Timer timer;
    extrapolateVelocityToAir(currentCfl);
    JET_INFO << "velocity extrapolation took "
             << timer.durationInSeconds() << " seconds";

    GridFluidSolver3::computeAdvection(timeIntervalInSeconds);
}

ScalarField3Ptr LevelSetLiquidSolver3::fluidSdf() const {
    return signedDistanceField();
}

void LevelSetLiquidSolver3::reinitialize(double currentCfl) {
    if (_levelSetSolver != nullptr) {
        auto sdf = signedDistanceField();
        auto sdf0 = sdf->clone();

        const Vector3D gridSpacing = sdf->gridSpacing();
        const double h = max3(gridSpacing.x, gridSpacing.y, gridSpacing.z);
        const double maxReinitDist
            = std::max(2.0 * currentCfl, _minReinitializeDistance) * h;

        _levelSetSolver->reinitialize(
            *sdf0, maxReinitDist, sdf.get());
        extrapolateIntoCollider(sdf.get());
    }
}

void LevelSetLiquidSolver3::extrapolateVelocityToAir(double currentCfl) {
    auto sdf = signedDistanceField();
    auto vel = gridSystemData()->velocity();

    auto u = vel->uAccessor();
    auto v = vel->vAccessor();
    auto w = vel->wAccessor();
    auto uPos = vel->uPosition();
    auto vPos = vel->vPosition();
    auto wPos = vel->wPosition();

    Array3<char> uMarker(u.size());
    Array3<char> vMarker(v.size());
    Array3<char> wMarker(w.size());

    uMarker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(sdf->sample(uPos(i, j, k)))) {
            uMarker(i, j, k) = 1;
        } else {
            uMarker(i, j, k) = 0;
            u(i, j, k) = 0;
        }
    });

    vMarker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(sdf->sample(vPos(i, j, k)))) {
            vMarker(i, j, k) = 1;
        } else {
            vMarker(i, j, k) = 0;
            v(i, j, k) = 0;
        }
    });

    wMarker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(sdf->sample(wPos(i, j, k)))) {
            wMarker(i, j, k) = 1;
        } else {
            wMarker(i, j, k) = 0;
            w(i, j, k) = 0;
        }
    });

    const Vector3D gridSpacing = sdf->gridSpacing();
    const double h = max3(gridSpacing.x, gridSpacing.y, gridSpacing.z);
    const double maxDist
        = std::max(2.0 * currentCfl, _minReinitializeDistance) * h;

    JET_INFO << "Max velocity extrapolation distance: " << maxDist;

    FmmLevelSetSolver3 fmmSolver;
    fmmSolver.extrapolate(*vel, *sdf, maxDist, vel.get());

    applyBoundaryCondition();
}

void LevelSetLiquidSolver3::addVolume(double volDiff) {
    auto sdf = signedDistanceField();
    const Vector3D gridSpacing = sdf->gridSpacing();
    const double cellVolume = gridSpacing.x * gridSpacing.y * gridSpacing.z;
    const double h = max3(gridSpacing.x, gridSpacing.y, gridSpacing.z);

    double volume0 = 0.0;
    double volume1 = 0.0;
    sdf->forEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        volume0 += 1.0 - smearedHeavisideSdf((*sdf)(i, j, k) / h);
        volume1 += 1.0 - smearedHeavisideSdf((*sdf)(i, j, k) / h + 1.0);
    });
    volume0 *= cellVolume;
    volume1 *= cellVolume;

    const double dVdh = (volume1 - volume0) / h;

    if (std::abs(dVdh) > 0.0) {
        double dist = volDiff / dVdh;

        sdf->parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
            (*sdf)(i, j, k) += dist;
        });
    }
}

LevelSetLiquidSolver3::Builder LevelSetLiquidSolver3::builder() {
    return Builder();
}


LevelSetLiquidSolver3 LevelSetLiquidSolver3::Builder::build() const {
    return LevelSetLiquidSolver3(
        _resolution,
        getGridSpacing(),
        _gridOrigin);
}

LevelSetLiquidSolver3Ptr LevelSetLiquidSolver3::Builder::makeShared() const {
    return std::shared_ptr<LevelSetLiquidSolver3>(
        new LevelSetLiquidSolver3(
            _resolution,
            getGridSpacing(),
            _gridOrigin),
        [] (LevelSetLiquidSolver3* obj) {
            delete obj;
        });
}
