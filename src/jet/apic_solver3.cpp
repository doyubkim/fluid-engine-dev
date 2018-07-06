// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/apic_solver3.h>
#include <pch.h>

using namespace jet;

ApicSolver3::ApicSolver3() : ApicSolver3({1, 1, 1}, {1, 1, 1}, {0, 0, 0}) {}

ApicSolver3::ApicSolver3(const Vector3UZ& resolution,
                         const Vector3D& gridSpacing,
                         const Vector3D& gridOrigin)
    : PicSolver3(resolution, gridSpacing, gridOrigin) {}

ApicSolver3::~ApicSolver3() {}

void ApicSolver3::transferFromParticlesToGrids() {
    auto flow = gridSystemData()->velocity();
    const auto particles = particleSystemData();
    const auto positions = particles->positions();
    auto velocities = particles->velocities();
    const size_t numberOfParticles = particles->numberOfParticles();
    const Vector3D hh = flow->gridSpacing() / 2.0;
    const auto bbox = flow->boundingBox();

    // Allocate buffers
    _cX.resize(numberOfParticles);
    _cY.resize(numberOfParticles);
    _cZ.resize(numberOfParticles);

    // Clear velocity to zero
    flow->fill(Vector3D());

    // Weighted-average velocity
    auto u = flow->uView();
    auto v = flow->vView();
    auto w = flow->wView();
    const auto uPos = flow->uPosition();
    const auto vPos = flow->vPosition();
    const auto wPos = flow->wPosition();
    Array3<double> uWeight(u.size());
    Array3<double> vWeight(v.size());
    Array3<double> wWeight(w.size());
    _uMarkers.resize(u.size());
    _vMarkers.resize(v.size());
    _wMarkers.resize(w.size());
    _uMarkers.fill(0);
    _vMarkers.fill(0);
    _wMarkers.fill(0);
    LinearArraySampler3<double> uSampler(flow->uView(), flow->gridSpacing(),
                                         flow->uOrigin());
    LinearArraySampler3<double> vSampler(flow->vView(), flow->gridSpacing(),
                                         flow->vOrigin());
    LinearArraySampler3<double> wSampler(flow->wView(), flow->gridSpacing(),
                                         flow->wOrigin());

    for (size_t i = 0; i < numberOfParticles; ++i) {
        std::array<Vector3UZ, 8> indices;
        std::array<double, 8> weights;

        auto uPosClamped = positions[i];
        uPosClamped.y = clamp(uPosClamped.y, bbox.lowerCorner.y + hh.y,
                              bbox.upperCorner.y - hh.y);
        uPosClamped.z = clamp(uPosClamped.z, bbox.lowerCorner.z + hh.z,
                              bbox.upperCorner.z - hh.z);
        uSampler.getCoordinatesAndWeights(uPosClamped, indices, weights);
        for (int j = 0; j < 8; ++j) {
            Vector3D gridPos = uPos(indices[j].x, indices[j].y, indices[j].z);
            double apicTerm = _cX[i].dot(gridPos - uPosClamped);
            u(indices[j]) += weights[j] * (velocities[i].x + apicTerm);
            uWeight(indices[j]) += weights[j];
            _uMarkers(indices[j]) = 1;
        }

        auto vPosClamped = positions[i];
        vPosClamped.x = clamp(vPosClamped.x, bbox.lowerCorner.x + hh.x,
                              bbox.upperCorner.x - hh.x);
        vPosClamped.z = clamp(vPosClamped.z, bbox.lowerCorner.z + hh.z,
                              bbox.upperCorner.z - hh.z);
        vSampler.getCoordinatesAndWeights(vPosClamped, indices, weights);
        for (int j = 0; j < 8; ++j) {
            Vector3D gridPos = vPos(indices[j].x, indices[j].y, indices[j].z);
            double apicTerm = _cY[i].dot(gridPos - vPosClamped);
            v(indices[j]) += weights[j] * (velocities[i].y + apicTerm);
            vWeight(indices[j]) += weights[j];
            _vMarkers(indices[j]) = 1;
        }

        auto wPosClamped = positions[i];
        wPosClamped.x = clamp(wPosClamped.x, bbox.lowerCorner.x + hh.x,
                              bbox.upperCorner.x - hh.x);
        wPosClamped.y = clamp(wPosClamped.y, bbox.lowerCorner.y + hh.y,
                              bbox.upperCorner.y - hh.y);
        wSampler.getCoordinatesAndWeights(wPosClamped, indices, weights);
        for (int j = 0; j < 8; ++j) {
            Vector3D gridPos = wPos(indices[j].x, indices[j].y, indices[j].z);
            double apicTerm = _cZ[i].dot(gridPos - wPosClamped);
            w(indices[j]) += weights[j] * (velocities[i].z + apicTerm);
            wWeight(indices[j]) += weights[j];
            _wMarkers(indices[j]) = 1;
        }
    }

    parallelForEachIndex(uWeight.size(), [&](size_t i, size_t j, size_t k) {
        if (uWeight(i, j, k) > 0.0) {
            u(i, j, k) /= uWeight(i, j, k);
        }
    });
    parallelForEachIndex(vWeight.size(), [&](size_t i, size_t j, size_t k) {
        if (vWeight(i, j, k) > 0.0) {
            v(i, j, k) /= vWeight(i, j, k);
        }
    });
    parallelForEachIndex(wWeight.size(), [&](size_t i, size_t j, size_t k) {
        if (wWeight(i, j, k) > 0.0) {
            w(i, j, k) /= wWeight(i, j, k);
        }
    });
}

void ApicSolver3::transferFromGridsToParticles() {
    const auto flow = gridSystemData()->velocity();
    const auto particles = particleSystemData();
    auto positions = particles->positions();
    auto velocities = particles->velocities();
    const size_t numberOfParticles = particles->numberOfParticles();
    const Vector3D hh = flow->gridSpacing() / 2.0;
    const auto bbox = flow->boundingBox();

    // Allocate buffers
    _cX.resize(numberOfParticles);
    _cY.resize(numberOfParticles);
    _cZ.resize(numberOfParticles);
    _cX.fill(Vector3D{});
    _cY.fill(Vector3D{});
    _cZ.fill(Vector3D{});

    auto u = flow->uView();
    auto v = flow->vView();
    auto w = flow->wView();
    LinearArraySampler3<double> uSampler(u, flow->gridSpacing(),
                                         flow->uOrigin());
    LinearArraySampler3<double> vSampler(v, flow->gridSpacing(),
                                         flow->vOrigin());
    LinearArraySampler3<double> wSampler(w, flow->gridSpacing(),
                                         flow->wOrigin());

    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        velocities[i] = flow->sample(positions[i]);

        std::array<Vector3UZ, 8> indices;
        std::array<Vector3D, 8> gradWeights;

        // x
        auto uPosClamped = positions[i];
        uPosClamped.y = clamp(uPosClamped.y, bbox.lowerCorner.y + hh.y,
                              bbox.upperCorner.y - hh.y);
        uPosClamped.z = clamp(uPosClamped.z, bbox.lowerCorner.z + hh.z,
                              bbox.upperCorner.z - hh.z);
        uSampler.getCoordinatesAndGradientWeights(uPosClamped, indices,
                                                  gradWeights);
        for (int j = 0; j < 8; ++j) {
            _cX[i] += gradWeights[j] * u(indices[j]);
        }

        // y
        auto vPosClamped = positions[i];
        vPosClamped.x = clamp(vPosClamped.x, bbox.lowerCorner.x + hh.x,
                              bbox.upperCorner.x - hh.x);
        vPosClamped.z = clamp(vPosClamped.z, bbox.lowerCorner.z + hh.z,
                              bbox.upperCorner.z - hh.z);
        vSampler.getCoordinatesAndGradientWeights(vPosClamped, indices,
                                                  gradWeights);
        for (int j = 0; j < 8; ++j) {
            _cY[i] += gradWeights[j] * v(indices[j]);
        }

        // z
        auto wPosClamped = positions[i];
        wPosClamped.x = clamp(wPosClamped.x, bbox.lowerCorner.x + hh.x,
                              bbox.upperCorner.x - hh.x);
        wPosClamped.y = clamp(wPosClamped.y, bbox.lowerCorner.y + hh.y,
                              bbox.upperCorner.y - hh.y);
        wSampler.getCoordinatesAndGradientWeights(wPosClamped, indices,
                                                  gradWeights);
        for (int j = 0; j < 8; ++j) {
            _cZ[i] += gradWeights[j] * w(indices[j]);
        }
    });
}

ApicSolver3::Builder ApicSolver3::builder() { return Builder(); }

ApicSolver3 ApicSolver3::Builder::build() const {
    return ApicSolver3(_resolution, getGridSpacing(), _gridOrigin);
}

ApicSolver3Ptr ApicSolver3::Builder::makeShared() const {
    return std::shared_ptr<ApicSolver3>(
        new ApicSolver3(_resolution, getGridSpacing(), _gridOrigin),
        [](ApicSolver3* obj) { delete obj; });
}
