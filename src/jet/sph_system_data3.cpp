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
#include <generated/sph_system_data3_generated.h>

#include <jet/bcc_lattice_point_generator.h>
#include <jet/parallel.h>
#include <jet/sph_kernels3.h>
#include <jet/sph_system_data3.h>

#include <algorithm>
#include <vector>

namespace jet {

SphSystemData3::SphSystemData3() : SphSystemData3(0) {}

SphSystemData3::SphSystemData3(size_t numberOfParticles)
    : ParticleSystemData3(numberOfParticles) {
    _densityIdx = addScalarData();
    _pressureIdx = addScalarData();

    setTargetSpacing(_targetSpacing);
}

SphSystemData3::SphSystemData3(const SphSystemData3& other) { set(other); }

SphSystemData3::~SphSystemData3() {}

void SphSystemData3::setRadius(double newRadius) {
    // Interpret it as setting target spacing
    setTargetSpacing(newRadius);
}

void SphSystemData3::setMass(double newMass) {
    double incRatio = newMass / mass();
    _targetDensity *= incRatio;
    ParticleSystemData3::setMass(newMass);
}

ConstArrayAccessor1<double> SphSystemData3::densities() const {
    return scalarDataAt(_densityIdx);
}

ArrayAccessor1<double> SphSystemData3::densities() {
    return scalarDataAt(_densityIdx);
}

ConstArrayAccessor1<double> SphSystemData3::pressures() const {
    return scalarDataAt(_pressureIdx);
}

ArrayAccessor1<double> SphSystemData3::pressures() {
    return scalarDataAt(_pressureIdx);
}

void SphSystemData3::updateDensities() {
    auto p = positions();
    auto d = densities();
    const double m = mass();

    parallelFor(kZeroSize, numberOfParticles(), [&](size_t i) {
        double sum = sumOfKernelNearby(p[i]);
        d[i] = m * sum;
    });
}

void SphSystemData3::setTargetDensity(double targetDensity) {
    _targetDensity = targetDensity;

    computeMass();
}

double SphSystemData3::targetDensity() const { return _targetDensity; }

void SphSystemData3::setTargetSpacing(double spacing) {
    ParticleSystemData3::setRadius(spacing);

    _targetSpacing = spacing;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

double SphSystemData3::targetSpacing() const { return _targetSpacing; }

void SphSystemData3::setRelativeKernelRadius(double relativeRadius) {
    _kernelRadiusOverTargetSpacing = relativeRadius;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

double SphSystemData3::relativeKernelRadius() const {
    return _kernelRadiusOverTargetSpacing;
}

void SphSystemData3::setKernelRadius(double kernelRadius) {
    _kernelRadius = kernelRadius;
    _targetSpacing = kernelRadius / _kernelRadiusOverTargetSpacing;

    computeMass();
}

double SphSystemData3::kernelRadius() const { return _kernelRadius; }

double SphSystemData3::sumOfKernelNearby(const Vector3D& origin) const {
    double sum = 0.0;
    SphStdKernel3 kernel(_kernelRadius);
    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius, [&](size_t, const Vector3D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            sum += kernel(dist);
        });
    return sum;
}

double SphSystemData3::interpolate(
    const Vector3D& origin, const ConstArrayAccessor1<double>& values) const {
    double sum = 0.0;
    auto d = densities();
    SphStdKernel3 kernel(_kernelRadius);
    const double m = mass();

    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius, [&](size_t i, const Vector3D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = m / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

Vector3D SphSystemData3::interpolate(
    const Vector3D& origin, const ConstArrayAccessor1<Vector3D>& values) const {
    Vector3D sum;
    auto d = densities();
    SphStdKernel3 kernel(_kernelRadius);
    const double m = mass();

    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius, [&](size_t i, const Vector3D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = m / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

Vector3D SphSystemData3::gradientAt(
    size_t i, const ConstArrayAccessor1<double>& values) const {
    Vector3D sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector3D origin = p[i];
    SphSpikyKernel3 kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector3D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        if (dist > 0.0) {
            Vector3D dir = (neighborPosition - origin) / dist;
            sum += d[i] * m *
                   (values[i] / square(d[i]) + values[j] / square(d[j])) *
                   kernel.gradient(dist, dir);
        }
    }

    return sum;
}

double SphSystemData3::laplacianAt(
    size_t i, const ConstArrayAccessor1<double>& values) const {
    double sum = 0.0;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector3D origin = p[i];
    SphSpikyKernel3 kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector3D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum +=
            m * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

Vector3D SphSystemData3::laplacianAt(
    size_t i, const ConstArrayAccessor1<Vector3D>& values) const {
    Vector3D sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector3D origin = p[i];
    SphSpikyKernel3 kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector3D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum +=
            m * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

void SphSystemData3::buildNeighborSearcher() {
    ParticleSystemData3::buildNeighborSearcher(_kernelRadius);
}

void SphSystemData3::buildNeighborLists() {
    ParticleSystemData3::buildNeighborLists(_kernelRadius);
}

void SphSystemData3::computeMass() {
    Array1<Vector3D> points;
    BccLatticePointGenerator pointsGenerator;
    BoundingBox3D sampleBound(
        Vector3D(-1.5 * _kernelRadius, -1.5 * _kernelRadius,
                 -1.5 * _kernelRadius),
        Vector3D(1.5 * _kernelRadius, 1.5 * _kernelRadius,
                 1.5 * _kernelRadius));

    pointsGenerator.generate(sampleBound, _targetSpacing, &points);

    double maxNumberDensity = 0.0;
    SphStdKernel3 kernel(_kernelRadius);

    for (size_t i = 0; i < points.size(); ++i) {
        const Vector3D& point = points[i];
        double sum = 0.0;

        for (size_t j = 0; j < points.size(); ++j) {
            const Vector3D& neighborPoint = points[j];
            sum += kernel(neighborPoint.distanceTo(point));
        }

        maxNumberDensity = std::max(maxNumberDensity, sum);
    }

    JET_ASSERT(maxNumberDensity > 0);

    double newMass = _targetDensity / maxNumberDensity;

    ParticleSystemData3::setMass(newMass);
}

void SphSystemData3::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);
    flatbuffers::Offset<fbs::ParticleSystemData3> fbsParticleSystemData;

    serializeParticleSystemData(&builder, &fbsParticleSystemData);

    auto fbsSphSystemData = fbs::CreateSphSystemData3(
        builder, fbsParticleSystemData, _targetDensity, _targetSpacing,
        _kernelRadiusOverTargetSpacing, _kernelRadius, _pressureIdx,
        _densityIdx);

    builder.Finish(fbsSphSystemData);

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void SphSystemData3::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsSphSystemData = fbs::GetSphSystemData3(buffer.data());

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

void SphSystemData3::set(const SphSystemData3& other) {
    ParticleSystemData3::set(other);

    _targetDensity = other._targetDensity;
    _targetSpacing = other._targetSpacing;
    _kernelRadiusOverTargetSpacing = other._kernelRadiusOverTargetSpacing;
    _kernelRadius = other._kernelRadius;
    _densityIdx = other._densityIdx;
    _pressureIdx = other._pressureIdx;
}

SphSystemData3& SphSystemData3::operator=(const SphSystemData3& other) {
    set(other);
    return *this;
}

}  // namespace jet
