// Copyright (c) 2016 Doyub Kim
//
// Adopted from the sample code of:
// Bart Adams and Martin Wicke,
// "Meshless Approximation Methods and Applications in Physics Based Modeling
// and Animation", Eurographics 2009 Tutorial

#include <pch.h>
#include <jet/bcc_lattice_point_generator.h>
#include <jet/parallel.h>
#include <jet/sph_kernels3.h>
#include <jet/sph_system_data3.h>
#include <algorithm>

namespace jet {

SphSystemData3::SphSystemData3() {
    _densityDataId = addScalarData();
    _pressureDataId = addScalarData();

    setTargetSpacing(_targetSpacing);
}

SphSystemData3::~SphSystemData3() {
}

void SphSystemData3::setRadius(double newRadius) {
    // Interprete it as setting target spacing
    setTargetSpacing(newRadius);
}

ConstArrayAccessor1<double> SphSystemData3::densities() const {
    return scalarDataAt(_densityDataId);
}

ArrayAccessor1<double> SphSystemData3::densities() {
    return scalarDataAt(_densityDataId);
}

ConstArrayAccessor1<double> SphSystemData3::pressures() const {
    return scalarDataAt(_pressureDataId);
}

ArrayAccessor1<double> SphSystemData3::pressures() {
    return scalarDataAt(_pressureDataId);
}

void SphSystemData3::updateDensities() {
    auto p = positions();
    auto d = densities();

    parallelFor(
        kZeroSize,
        numberOfParticles(),
        [&](size_t i) {
            double sum = sumOfKernelNearby(p[i]);
            d[i] = _mass * sum;
        });
}

void SphSystemData3::setTargetDensity(double targetDensity) {
    _targetDensity = targetDensity;

    computeMass();
}

double SphSystemData3::targetDensity() const {
    return _targetDensity;
}

double SphSystemData3::mass() const {
    return _mass;
}

void SphSystemData3::setTargetSpacing(double spacing) {
    ParticleSystemData3::setRadius(spacing);

    _targetSpacing = spacing;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

double SphSystemData3::targetSpacing() const {
    return _targetSpacing;
}

void SphSystemData3::setRelativeKernelRadius(double relativeRadius) {
    _kernelRadiusOverTargetSpacing = relativeRadius;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

double SphSystemData3::relativeKernelRadius() const {
    return _kernelRadiusOverTargetSpacing;
}

double SphSystemData3::kernelRadius() const {
    return _kernelRadius;
}

double SphSystemData3::sumOfKernelNearby(const Vector3D& origin) const {
    double sum = 0.0;
    SphStdKernel3 kernel(_kernelRadius);
    neighborSearcher()->forEachNearbyPoint(
        origin,
        _kernelRadius,
        [&] (size_t, const Vector3D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            sum += kernel(dist);
        });
    return sum;
}

double SphSystemData3::interpolate(
    const Vector3D& origin,
    const ConstArrayAccessor1<double>& values) const {
    double sum = 0.0;
    auto d = densities();
    SphStdKernel3 kernel(_kernelRadius);

    neighborSearcher()->forEachNearbyPoint(
        origin,
        _kernelRadius,
        [&] (size_t i, const Vector3D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = _mass / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

Vector3D SphSystemData3::interpolate(
    const Vector3D& origin,
    const ConstArrayAccessor1<Vector3D>& values) const {
    Vector3D sum;
    auto d = densities();
    SphStdKernel3 kernel(_kernelRadius);

    neighborSearcher()->forEachNearbyPoint(
        origin,
        _kernelRadius,
        [&] (size_t i, const Vector3D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = _mass / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

Vector3D SphSystemData3::gradientAt(
    size_t i,
    const ConstArrayAccessor1<double>& values) const {
    Vector3D sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector3D origin = p[i];
    SphSpikyKernel3 kernel(_kernelRadius);

    for (size_t j : neighbors) {
        Vector3D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        if (dist > 0.0) {
            Vector3D dir = (neighborPosition - origin) / dist;
            sum
                += d[i] * _mass
                * (values[i] / square(d[i]) + values[j] / square(d[j]))
                * kernel.gradient(dist, dir);
        }
    }

    return sum;
}

double SphSystemData3::laplacianAt(
    size_t i,
    const ConstArrayAccessor1<double>& values) const {
    double sum = 0.0;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector3D origin = p[i];
    SphSpikyKernel3 kernel(_kernelRadius);

    for (size_t j : neighbors) {
        Vector3D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum += _mass
            * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

Vector3D SphSystemData3::laplacianAt(
    size_t i,
    const ConstArrayAccessor1<Vector3D>& values) const {
    Vector3D sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector3D origin = p[i];
    SphSpikyKernel3 kernel(_kernelRadius);

    for (size_t j : neighbors) {
        Vector3D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum += _mass
            * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

void SphSystemData3::buildNeighborSearcher() {
    ParticleSystemData3::buildNeighborSearcher(_kernelRadius);
}

void SphSystemData3::buildNeighborLists() {
    ParticleSystemData3::buildNeighborLists(_kernelRadius);
}

void SphSystemData3::setMass(double newMass) {
    // Ignore input
    UNUSED_VARIABLE(newMass);
}

void SphSystemData3::computeMass() {
    Array1<Vector3D> points;
    BccLatticePointGenerator pointsGenerator;
    BoundingBox3D sampleBound(
        Vector3D(-1.5*_kernelRadius, -1.5*_kernelRadius, -1.5*_kernelRadius),
        Vector3D(1.5*_kernelRadius, 1.5*_kernelRadius, 1.5*_kernelRadius));

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

    _mass = _targetDensity / maxNumberDensity;

    ParticleSystemData3::setMass(_mass);
}

}  // namespace jet
