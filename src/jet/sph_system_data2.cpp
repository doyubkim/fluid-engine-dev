// Copyright (c) 2016 Doyub Kim
//
// Adopted from the sample code of:
// Bart Adams and Martin Wicke,
// "Meshless Approximation Methods and Applications in Physics Based Modeling
// and Animation", Eurographics 2009 Tutorial

#include <pch.h>
#include <jet/parallel.h>
#include <jet/sph_kernels2.h>
#include <jet/sph_system_data2.h>
#include <jet/triangle_point_generator.h>
#include <algorithm>

namespace jet {

SphSystemData2::SphSystemData2() {
    _densityDataId = addScalarData();
    _pressureDataId = addScalarData();

    setTargetSpacing(_targetSpacing);
}

SphSystemData2::~SphSystemData2() {
}

void SphSystemData2::setRadius(double newRadius) {
    // Interprete it as setting target spacing
    setTargetSpacing(newRadius);
}

ConstArrayAccessor1<double> SphSystemData2::densities() const {
    return scalarDataAt(_densityDataId);
}

ArrayAccessor1<double> SphSystemData2::densities() {
    return scalarDataAt(_densityDataId);
}

ConstArrayAccessor1<double> SphSystemData2::pressures() const {
    return scalarDataAt(_pressureDataId);
}

ArrayAccessor1<double> SphSystemData2::pressures() {
    return scalarDataAt(_pressureDataId);
}

void SphSystemData2::updateDensities() {
    auto p = positions();
    auto d = densities();

    parallelFor(
        kZeroSize,
        numberOfParticles(),
        [&] (size_t i) {
            double sum = sumOfKernelNearby(p[i]);
            d[i] = _mass * sum;
        });
}

void SphSystemData2::setTargetDensity(double targetDensity) {
    _targetDensity = targetDensity;

    computeMass();
}

double SphSystemData2::targetDensity() const {
    return _targetDensity;
}

double SphSystemData2::mass() const {
    return _mass;
}

void SphSystemData2::setTargetSpacing(double spacing) {
    ParticleSystemData2::setRadius(spacing);

    _targetSpacing = spacing;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

double SphSystemData2::targetSpacing() const {
    return _targetSpacing;
}

void SphSystemData2::setRelativeKernelRadius(double relativeRadius) {
    _kernelRadiusOverTargetSpacing = relativeRadius;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

double SphSystemData2::relativeKernelRadius() const {
    return _kernelRadiusOverTargetSpacing;
}

double SphSystemData2::kernelRadius() const {
    return _kernelRadius;
}

double SphSystemData2::sumOfKernelNearby(const Vector2D& origin) const {
    double sum = 0.0;
    SphStdKernel2 kernel(_kernelRadius);
    neighborSearcher()->forEachNearbyPoint(
        origin,
        _kernelRadius,
        [&] (size_t, const Vector2D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            sum += kernel(dist);
        });
    return sum;
}

double SphSystemData2::interpolate(
    const Vector2D& origin,
    const ConstArrayAccessor1<double>& values) const {
    double sum = 0.0;
    auto d = densities();
    SphStdKernel2 kernel(_kernelRadius);

    neighborSearcher()->forEachNearbyPoint(
        origin,
        _kernelRadius,
        [&] (size_t i, const Vector2D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = _mass / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

Vector2D SphSystemData2::interpolate(
    const Vector2D& origin,
    const ConstArrayAccessor1<Vector2D>& values) const {
    Vector2D sum;
    auto d = densities();
    SphStdKernel2 kernel(_kernelRadius);

    neighborSearcher()->forEachNearbyPoint(
        origin,
        _kernelRadius,
        [&] (size_t i, const Vector2D& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = _mass / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

Vector2D SphSystemData2::gradientAt(
    size_t i,
    const ConstArrayAccessor1<double>& values) const {
    Vector2D sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector2D origin = p[i];
    SphSpikyKernel2 kernel(_kernelRadius);

    for (size_t j : neighbors) {
        Vector2D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        if (dist > 0.0) {
            Vector2D dir = (neighborPosition - origin) / dist;
            sum
                += d[i] * _mass
                * (values[i] / square(d[i]) + values[j] / square(d[j]))
                * kernel.gradient(dist, dir);
        }
    }

    return sum;
}

double SphSystemData2::laplacianAt(
    size_t i,
    const ConstArrayAccessor1<double>& values) const {
    double sum = 0.0;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector2D origin = p[i];
    SphSpikyKernel2 kernel(_kernelRadius);

    for (size_t j : neighbors) {
        Vector2D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum += _mass
            * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

Vector2D SphSystemData2::laplacianAt(
    size_t i,
    const ConstArrayAccessor1<Vector2D>& values) const {
    Vector2D sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector2D origin = p[i];
    SphSpikyKernel2 kernel(_kernelRadius);

    for (size_t j : neighbors) {
        Vector2D neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum += _mass
            * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

void SphSystemData2::buildNeighborSearcher() {
    ParticleSystemData2::buildNeighborSearcher(_kernelRadius);
}

void SphSystemData2::buildNeighborLists() {
    ParticleSystemData2::buildNeighborLists(_kernelRadius);
}

void SphSystemData2::setMass(double newMass) {
    // Ignore input
    UNUSED_VARIABLE(newMass);
}

void SphSystemData2::computeMass() {
    Array1<Vector2D> points;
    TrianglePointGenerator pointsGenerator;
    BoundingBox2D sampleBound(
        Vector2D(-1.5*_kernelRadius, -1.5*_kernelRadius),
        Vector2D(1.5*_kernelRadius, 1.5*_kernelRadius));

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

    _mass = _targetDensity / maxNumberDensity;

    ParticleSystemData2::setMass(_mass);
}

}  // namespace jet
