// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/flip_solver3.h>

using namespace jet;

FlipSolver3::FlipSolver3() : FlipSolver3({1, 1, 1}, {1, 1, 1}, {0, 0, 0}) {
}

FlipSolver3::FlipSolver3(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& gridOrigin)
: PicSolver3(resolution, gridSpacing, gridOrigin) {
}

FlipSolver3::~FlipSolver3() {
}

void FlipSolver3::transferFromParticlesToGrids() {
    PicSolver3::transferFromParticlesToGrids();

    // Store snapshot
    _delta.set(*gridSystemData()->velocity());
}

void FlipSolver3::transferFromGridsToParticles() {
    auto flow = gridSystemData()->velocity();
    auto positions = particleSystemData()->positions();
    auto velocities = particleSystemData()->velocities();
    size_t numberOfParticles = particleSystemData()->numberOfParticles();

    // Compute delta
    flow->parallelForEachUIndex([&](size_t i, size_t j, size_t k) {
        _delta.u(i, j, k) = flow->u(i, j, k) - _delta.u(i, j, k);
    });

    flow->parallelForEachVIndex([&](size_t i, size_t j, size_t k) {
        _delta.v(i, j, k) = flow->v(i, j, k) - _delta.v(i, j, k);
    });

    flow->parallelForEachWIndex([&](size_t i, size_t j, size_t k) {
        _delta.w(i, j, k) = flow->w(i, j, k) - _delta.w(i, j, k);
    });

    // Transfer delta to the particles
    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        velocities[i] += _delta.sample(positions[i]);
    });
}

FlipSolver3::Builder FlipSolver3::builder() {
    return Builder();
}


FlipSolver3 FlipSolver3::Builder::build() const {
    return FlipSolver3(_resolution, getGridSpacing(), _gridOrigin);
}

FlipSolver3Ptr FlipSolver3::Builder::makeShared() const {
    return std::shared_ptr<FlipSolver3>(
        new FlipSolver3(
            _resolution,
            getGridSpacing(),
            _gridOrigin),
        [] (FlipSolver3* obj) {
            delete obj;
        });
}
