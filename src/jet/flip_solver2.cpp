// Copyright (c) 2017 Doyub Kim

#include <pch.h>
#include <jet/flip_solver2.h>

using namespace jet;

FlipSolver2::FlipSolver2() : FlipSolver2({1, 1}, {1, 1}, {0, 0}) {
}

FlipSolver2::FlipSolver2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin)
: PicSolver2(resolution, gridSpacing, gridOrigin) {
}

FlipSolver2::~FlipSolver2() {
}

void FlipSolver2::transferFromParticlesToGrids() {
    PicSolver2::transferFromParticlesToGrids();

    // Store snapshot
    auto vel = gridSystemData()->velocity();
    auto u = gridSystemData()->velocity()->uConstAccessor();
    auto v = gridSystemData()->velocity()->vConstAccessor();
    _uDelta.resize(u.size());
    _vDelta.resize(v.size());

    vel->parallelForEachUIndex([&](size_t i, size_t j) {
        _uDelta(i, j) = static_cast<float>(u(i, j));
    });
    vel->parallelForEachVIndex([&](size_t i, size_t j) {
        _vDelta(i, j) = static_cast<float>(v(i, j));
    });
}

void FlipSolver2::transferFromGridsToParticles() {
    auto flow = gridSystemData()->velocity();
    auto positions = particleSystemData()->positions();
    auto velocities = particleSystemData()->velocities();
    size_t numberOfParticles = particleSystemData()->numberOfParticles();

    // Compute delta
    flow->parallelForEachUIndex([&](size_t i, size_t j) {
        flow->u(i, j) -= _uDelta(i, j);
    });

    flow->parallelForEachVIndex([&](size_t i, size_t j) {
        flow->v(i, j) -= _vDelta(i, j);
    });

    // Transfer delta to the particles
    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        velocities[i] += flow->sample(positions[i]);
    });
}

FlipSolver2::Builder FlipSolver2::builder() {
    return Builder();
}


FlipSolver2 FlipSolver2::Builder::build() const {
    return FlipSolver2(_resolution, getGridSpacing(), _gridOrigin);
}

FlipSolver2Ptr FlipSolver2::Builder::makeShared() const {
    return std::shared_ptr<FlipSolver2>(
        new FlipSolver2(
            _resolution,
            getGridSpacing(),
            _gridOrigin),
        [] (FlipSolver2* obj) {
            delete obj;
        });
}
