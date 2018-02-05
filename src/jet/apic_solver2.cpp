// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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
    const auto particles = particleSystemData();
    const auto positions = particles->positions();
    auto velocities = particles->velocities();
    const size_t numberOfParticles = particles->numberOfParticles();
    const auto hh = flow->gridSpacing() / 2.0;
    const auto bbox = flow->boundingBox();

    // Allocate buffers
    _cX.resize(numberOfParticles);
    _cY.resize(numberOfParticles);

    // Clear velocity to zero
    flow->fill(Vector2D());

    // Weighted-average velocity
    auto u = flow->uAccessor();
    auto v = flow->vAccessor();
    const auto uPos = flow->uPosition();
    const auto vPos = flow->vPosition();
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

        auto uPosClamped = positions[i];
        uPosClamped.y = clamp(
            uPosClamped.y,
            bbox.lowerCorner.y + hh.y,
            bbox.upperCorner.y - hh.y);
        uSampler.getCoordinatesAndWeights(uPosClamped, &indices, &weights);
        for (int j = 0; j < 4; ++j) {
            Vector2D gridPos = uPos(indices[j].x, indices[j].y);
            double apicTerm = _cX[i].dot(gridPos - uPosClamped);
            u(indices[j]) += weights[j] * (velocities[i].x + apicTerm);
            uWeight(indices[j]) += weights[j];
            _uMarkers(indices[j]) = 1;
        }

        auto vPosClamped = positions[i];
        vPosClamped.x = clamp(
            vPosClamped.x,
            bbox.lowerCorner.x + hh.x,
            bbox.upperCorner.x - hh.x);
        vSampler.getCoordinatesAndWeights(vPosClamped, &indices, &weights);
        for (int j = 0; j < 4; ++j) {
            Vector2D gridPos = vPos(indices[j].x, indices[j].y);
            double apicTerm = _cY[i].dot(gridPos - vPosClamped);
            v(indices[j]) += weights[j] * (velocities[i].y + apicTerm);
            vWeight(indices[j]) += weights[j];
            _vMarkers(indices[j]) = 1;
        }
    }

    uWeight.parallelForEachIndex([&](size_t i, size_t j) {
        if (uWeight(i, j) > 0.0) {
            u(i, j) /= uWeight(i, j);
        }
    });
    vWeight.parallelForEachIndex([&](size_t i, size_t j) {
        if (vWeight(i, j) > 0.0) {
            v(i, j) /= vWeight(i, j);
        }
    });
}

void ApicSolver2::transferFromGridsToParticles() {
    const auto flow = gridSystemData()->velocity();
    auto particles = particleSystemData();
    auto positions = particles->positions();
    auto velocities = particles->velocities();
    const size_t numberOfParticles = particles->numberOfParticles();
    const auto hh = flow->gridSpacing() / 2.0;
    const auto bbox = flow->boundingBox();

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
        auto uPosClamped = positions[i];
        uPosClamped.y = clamp(
            uPosClamped.y,
            bbox.lowerCorner.y + hh.y,
            bbox.upperCorner.y - hh.y);
        uSampler.getCoordinatesAndGradientWeights(
            uPosClamped, &indices, &gradWeights);
        for (int j = 0; j < 4; ++j) {
            _cX[i] += gradWeights[j] * u(indices[j]);
        }

        // y
        auto vPosClamped = positions[i];
        vPosClamped.x = clamp(
            vPosClamped.x,
            bbox.lowerCorner.x + hh.x,
            bbox.upperCorner.x - hh.x);
        vSampler.getCoordinatesAndGradientWeights(
            vPosClamped, &indices, &gradWeights);
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
