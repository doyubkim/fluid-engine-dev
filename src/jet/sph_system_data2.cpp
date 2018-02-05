// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// Adopted from the sample code of:
// Bart Adams and Martin Wicke,
// "Meshless Approximation Methods and Applications in Physics Based Modeling
// and Animation", Eurographics 2009 Tutorial

#include <pch.h>

#include <fbs_helpers.h>
#include <generated/sph_system_data2_generated.h>

#include <jet/parallel.h>
#include <jet/sph_kernels2.h>
#include <jet/sph_system_data2.h>
#include <jet/triangle_point_generator.h>

#include <algorithm>
#include <vector>

namespace jet {

SphSystemData2::SphSystemData2() : SphSystemData2(0) {}

SphSystemData2::SphSystemData2(size_t numberOfParticles)
    : ParticleSystemData2(numberOfParticles) {
    _densityIdx = addScalarData();
    _pressureIdx = addScalarData();

    setTargetSpacing(_targetSpacing);
}

SphSystemData2::SphSystemData2(const SphSystemData2& other) { set(other); }

SphSystemData2::~SphSystemData2() {}

void SphSystemData2::setRadius(double newRadius) {
    // Interpret it as setting target spacing
    setTargetSpacing(newRadius);
}

void SphSystemData2::setMass(double newMass) {
    double incRatio = newMass / mass();
    _targetDensity *= incRatio;
    ParticleSystemData2::setMass(newMass);
}

ConstArrayAccessor1<double> SphSystemData2::densities() const {
    return scalarDataAt(_densityIdx);
}

ArrayAccessor1<double> SphSystemData2::densities() {
    return scalarDataAt(_densityIdx);
}

ConstArrayAccessor1<double> SphSystemData2::pressures() const {
    return scalarDataAt(_pressureIdx);
}

ArrayAccessor1<double> SphSystemData2::pressures() {
    return scalarDataAt(_pressureIdx);
}

void SphSystemData2::updateDensities() {
    auto p = positions();
    auto d = densities();
    const double m = mass();

    parallelFor(kZeroSize, numberOfParticles(), [&](size_t i) {
        double sum = sumOfKernelNearby(p[i]);
        d[i] = m * sum;
    });
}

void SphSystemData2::setTargetDensity(double targetDensity) {
    _targetDensity = targetDensity;

    computeMass();
}

double SphSystemData2::targetDensity() const { return _targetDensity; }

void SphSystemData2::setTargetSpacing(double spacing) {
    ParticleSystemData2::setRadius(spacing);

    _targetSpacing = spacing;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

double SphSystemData2::targetSpacing() const { return _targetSpacing; }

void SphSystemData2::setRelativeKernelRadius(double relativeRadius) {
    _kernelRadiusOverTargetSpacing = relativeRadius;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

double SphSystemData2::relativeKernelRadius() const {
    return _kernelRadiusOverTargetSpacing;
}

void SphSystemData2::setKernelRadius(double kernelRadius) {
    _kernelRadius = kernelRadius;
    _targetSpacing = kernelRadius / _kernelRadiusOverTargetSpacing;

    computeMass();
}

double SphSystemData2::kernelRadius() const { return _kernelRadius; }

double SphSystemData2::sumOfKernelNearby(const Vector2D& origin) const {
    double sum = 0.0;
    SphStdKernel2 kernel(_kernelRadius);
    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius, [&](size_t, const Vector2D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            sum += kernel(dist);
        });
    return sum;
}

double SphSystemData2::interpolate(
    const Vector2D& origin, const ConstArrayAccessor1<double>& values) const {
    double sum = 0.0;
    auto d = densities();
    SphStdKernel2 kernel(_kernelRadius);
    const double m = mass();

    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius, [&](size_t i, const Vector2D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = m / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

Vector2D SphSystemData2::interpolate(
    const Vector2D& origin, const ConstArrayAccessor1<Vector2D>& values) const {
    Vector2D sum;
    auto d = densities();
    SphStdKernel2 kernel(_kernelRadius);
    const double m = mass();

    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius, [&](size_t i, const Vector2D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = m / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

Vector2D SphSystemData2::gradientAt(
    size_t i, const ConstArrayAccessor1<double>& values) const {
    Vector2D sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector2D origin = p[i];
    SphSpikyKernel2 kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector2D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        if (dist > 0.0) {
            Vector2D dir = (neighborPosition - origin) / dist;
            sum += d[i] * m *
                   (values[i] / square(d[i]) + values[j] / square(d[j])) *
                   kernel.gradient(dist, dir);
        }
    }

    return sum;
}

double SphSystemData2::laplacianAt(
    size_t i, const ConstArrayAccessor1<double>& values) const {
    double sum = 0.0;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector2D origin = p[i];
    SphSpikyKernel2 kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector2D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum +=
            m * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

Vector2D SphSystemData2::laplacianAt(
    size_t i, const ConstArrayAccessor1<Vector2D>& values) const {
    Vector2D sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector2D origin = p[i];
    SphSpikyKernel2 kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector2D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum +=
            m * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

void SphSystemData2::buildNeighborSearcher() {
    ParticleSystemData2::buildNeighborSearcher(_kernelRadius);
}

void SphSystemData2::buildNeighborLists() {
    ParticleSystemData2::buildNeighborLists(_kernelRadius);
}

void SphSystemData2::computeMass() {
    Array1<Vector2D> points;
    TrianglePointGenerator pointsGenerator;
    BoundingBox2D sampleBound(
        Vector2D(-1.5 * _kernelRadius, -1.5 * _kernelRadius),
        Vector2D(1.5 * _kernelRadius, 1.5 * _kernelRadius));

    pointsGenerator.generate(sampleBound, _targetSpacing, &points);

    double maxNumberDensity = 0.0;
    SphStdKernel2 kernel(_kernelRadius);

    for (size_t i = 0; i < points.size(); ++i) {
        const Vector2D& point = points[i];
        double sum = 0.0;

        for (size_t j = 0; j < points.size(); ++j) {
            const Vector2D& neighborPoint = points[j];
            sum += kernel(neighborPoint.distanceTo(point));
        }

        maxNumberDensity = std::max(maxNumberDensity, sum);
    }

    JET_ASSERT(maxNumberDensity > 0);

    double newMass = _targetDensity / maxNumberDensity;

    ParticleSystemData2::setMass(newMass);
}

void SphSystemData2::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);
    flatbuffers::Offset<fbs::ParticleSystemData2> fbsParticleSystemData;

    serializeParticleSystemData(&builder, &fbsParticleSystemData);

    auto fbsSphSystemData = fbs::CreateSphSystemData2(
        builder, fbsParticleSystemData, _targetDensity, _targetSpacing,
        _kernelRadiusOverTargetSpacing, _kernelRadius, _pressureIdx,
        _densityIdx);

    builder.Finish(fbsSphSystemData);

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void SphSystemData2::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsSphSystemData = fbs::GetSphSystemData2(buffer.data());

    auto base = fbsSphSystemData->base();
    deserializeParticleSystemData(base);

    // SPH specific
    _targetDensity = fbsSphSystemData->targetDensity();
    _targetSpacing = fbsSphSystemData->targetSpacing();
    _kernelRadiusOverTargetSpacing =
        fbsSphSystemData->kernelRadiusOverTargetSpacing();
    _kernelRadius = fbsSphSystemData->kernelRadius();
    _pressureIdx = static_cast<size_t>(fbsSphSystemData->pressureIdx());
    _densityIdx = static_cast<size_t>(fbsSphSystemData->densityIdx());
}

void SphSystemData2::set(const SphSystemData2& other) {
    ParticleSystemData2::set(other);

    _targetDensity = other._targetDensity;
    _targetSpacing = other._targetSpacing;
    _kernelRadiusOverTargetSpacing = other._kernelRadiusOverTargetSpacing;
    _kernelRadius = other._kernelRadius;
    _densityIdx = other._densityIdx;
    _pressureIdx = other._pressureIdx;
}

SphSystemData2& SphSystemData2::operator=(const SphSystemData2& other) {
    set(other);
    return *this;
}

}  // namespace jet
