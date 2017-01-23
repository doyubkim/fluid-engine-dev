// Copyright (c) 2017 Doyub Kim

#include <pch.h>
#include <jet/apic_solver2.h>

using namespace jet;

ApicSolver2::ApicSolver2() : ApicSolver2({1, 1}, {1, 1}, {0, 0}) {
}

ApicSolver2::ApicSolver2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin)
: PicSolver2(resolution, gridSpacing, gridOrigin) {
}

ApicSolver2::~ApicSolver2() {
}

void ApicSolver2::transferFromParticlesToGrids() {
    auto flow = gridSystemData()->velocity();
    auto particles = particleSystemData();
    auto positions = particles->positions();
    auto velocities = particles->velocities();
    size_t numberOfParticles = particles->numberOfParticles();

    // Allocate buffers
    _cX.resize(numberOfParticles);
    _cY.resize(numberOfParticles);

    // Clear velocity to zero
    flow->fill(Vector2D());

    // Weighted-average velocity
    auto u = flow->uAccessor();
    auto v = flow->vAccessor();
    auto uPos = flow->uPosition();
    auto vPos = flow->vPosition();
    Array2<double> uWeight(u.size());
    Array2<double> vWeight(v.size());
    _uMarkers.resize(u.size());
    _vMarkers.resize(v.size());
    _uMarkers.set(0);
    _vMarkers.set(0);
    LinearArraySampler2<double, double> uSampler(
        flow->uConstAccessor(),
        flow->gridSpacing(),
        flow->uOrigin());
    LinearArraySampler2<double, double> vSampler(
        flow->vConstAccessor(),
        flow->gridSpacing(),
        flow->vOrigin());

    for (size_t i = 0; i < numberOfParticles; ++i) {
        std::array<Point2UI, 4> indices;
        std::array<double, 4> weights;

        uSampler.getCoordinatesAndWeights(positions[i], &indices, &weights);
        for (int j = 0; j < 4; ++j) {
            Vector2D gridPos = uPos(indices[j].x, indices[j].y);
            double apicTerm = _cX[i].dot(gridPos - positions[i]);
            u(indices[j]) += weights[j] * (velocities[i].x + apicTerm);
            uWeight(indices[j]) += weights[j];
            _uMarkers(indices[j]) = 1;
        }

        vSampler.getCoordinatesAndWeights(positions[i], &indices, &weights);
        for (int j = 0; j < 4; ++j) {
            Vector2D gridPos = vPos(indices[j].x, indices[j].y);
            double apicTerm = _cY[i].dot(gridPos - positions[i]);
            v(indices[j]) += weights[j] * (velocities[i].y + apicTerm);
            vWeight(indices[j]) += weights[j];
            _vMarkers(indices[j]) = 1;
        }
    }

    uWeight.forEachIndex([&](size_t i, size_t j) {
        if (uWeight(i, j) > 0.0) {
            u(i, j) /= uWeight(i, j);
        }
    });
    vWeight.forEachIndex([&](size_t i, size_t j) {
        if (vWeight(i, j) > 0.0) {
            v(i, j) /= vWeight(i, j);
        }
    });
}

void ApicSolver2::transferFromGridsToParticles() {
    auto flow = gridSystemData()->velocity();
    auto particles = particleSystemData();
    auto positions = particles->positions();
    auto velocities = particles->velocities();
    size_t numberOfParticles = particles->numberOfParticles();

    // Allocate buffers
    _cX.resize(numberOfParticles);
    _cY.resize(numberOfParticles);
    _cX.set(Vector2D());
    _cY.set(Vector2D());

    auto u = flow->uAccessor();
    auto v = flow->vAccessor();
    LinearArraySampler2<double, double> uSampler(
        u, flow->gridSpacing(), flow->uOrigin());
    LinearArraySampler2<double, double> vSampler(
        v, flow->gridSpacing(), flow->vOrigin());

    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        velocities[i] = flow->sample(positions[i]);

        std::array<Point2UI, 4> indices;
        std::array<Vector2D, 4> gradWeights;

        // x
        uSampler.getCoordinatesAndGradientWeights(
            positions[i], &indices, &gradWeights);
        for (int j = 0; j < 4; ++j) {
            _cX[i] += gradWeights[j] * u(indices[j]);
        }

        // y
        vSampler.getCoordinatesAndGradientWeights(
            positions[i], &indices, &gradWeights);
        for (int j = 0; j < 4; ++j) {
            _cY[i] += gradWeights[j] * v(indices[j]);
        }
    });
}

ApicSolver2::Builder ApicSolver2::builder() {
    return Builder();
}


ApicSolver2 ApicSolver2::Builder::build() const {
    return ApicSolver2(_resolution, getGridSpacing(), _gridOrigin);
}

ApicSolver2Ptr ApicSolver2::Builder::makeShared() const {
    return std::shared_ptr<ApicSolver2>(
        new ApicSolver2(
            _resolution,
            getGridSpacing(),
            _gridOrigin),
        [] (ApicSolver2* obj) {
            delete obj;
        });
}
