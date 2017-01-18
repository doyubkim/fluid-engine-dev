// Copyright (c) 2017 Doyub Kim

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
    auto vel = gridSystemData()->velocity();
    auto u = gridSystemData()->velocity()->uConstAccessor();
    auto v = gridSystemData()->velocity()->vConstAccessor();
    auto w = gridSystemData()->velocity()->wConstAccessor();
    _uDelta.resize(u.size());
    _vDelta.resize(v.size());
    _wDelta.resize(w.size());

    vel->parallelForEachUIndex([&](size_t i, size_t j, size_t k) {
        _uDelta(i, j, k) = static_cast<float>(u(i, j, k));
    });
    vel->parallelForEachVIndex([&](size_t i, size_t j, size_t k) {
        _vDelta(i, j, k) = static_cast<float>(v(i, j, k));
    });
    vel->parallelForEachWIndex([&](size_t i, size_t j, size_t k) {
        _wDelta(i, j, k) = static_cast<float>(w(i, j, k));
    });
}

void FlipSolver3::transferFromGridsToParticles() {
    auto flow = gridSystemData()->velocity();
    auto positions = particleSystemData()->positions();
    auto velocities = particleSystemData()->velocities();
    size_t numberOfParticles = particleSystemData()->numberOfParticles();

    // Compute delta
    flow->parallelForEachUIndex([&](size_t i, size_t j, size_t k) {
        _uDelta(i, j, k) = flow->u(i, j, k) - _uDelta(i, j, k);
    });

    flow->parallelForEachVIndex([&](size_t i, size_t j, size_t k) {
        _vDelta(i, j, k) = flow->v(i, j, k) - _vDelta(i, j, k);
    });

    flow->parallelForEachWIndex([&](size_t i, size_t j, size_t k) {
        _wDelta(i, j, k) = flow->w(i, j, k) - _wDelta(i, j, k);
    });

    LinearArraySampler3<float, float> uSampler(
        _uDelta.constAccessor(),
        flow->gridSpacing().castTo<float>(),
        flow->uOrigin().castTo<float>());
    LinearArraySampler3<float, float> vSampler(
        _vDelta.constAccessor(),
        flow->gridSpacing().castTo<float>(),
        flow->vOrigin().castTo<float>());
    LinearArraySampler3<float, float> wSampler(
        _wDelta.constAccessor(),
        flow->gridSpacing().castTo<float>(),
        flow->wOrigin().castTo<float>());

    auto sampler = [uSampler, vSampler, wSampler](const Vector3D& x) {
        const auto xf = x.castTo<float>();
        double u = uSampler(xf);
        double v = vSampler(xf);
        double w = wSampler(xf);
        return Vector3D(u, v, w);
    };

    // Transfer delta to the particles
    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        velocities[i] += sampler(positions[i]);
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
