// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_sph_system_data2_func.h"

#include <jet/cuda_sph_system_data2.h>
#include <jet/sph_kernels2.h>
#include <jet/triangle_point_generator.h>

#include <thrust/extrema.h>
#include <thrust/for_each.h>

using namespace jet;
using namespace experimental;

CudaSphSystemData2::CudaSphSystemData2() : CudaSphSystemData2(0) {}

CudaSphSystemData2::CudaSphSystemData2(size_t numberOfParticles)
    : CudaParticleSystemData2(numberOfParticles) {
    _densityIdx = addFloatData();
    _pressureIdx = addFloatData();

    setTargetSpacing(_targetSpacing);
}

CudaSphSystemData2::CudaSphSystemData2(const CudaSphSystemData2& other) {
    set(other);
}

CudaSphSystemData2::~CudaSphSystemData2() {}

const CudaArrayView1<float> CudaSphSystemData2::densities() const {
    return floatDataAt(_densityIdx);
}

CudaArrayView1<float> CudaSphSystemData2::densities() {
    return floatDataAt(_densityIdx);
}

const CudaArrayView1<float> CudaSphSystemData2::pressures() const {
    return floatDataAt(_pressureIdx);
}

CudaArrayView1<float> CudaSphSystemData2::pressures() {
    return floatDataAt(_pressureIdx);
}

void CudaSphSystemData2::updateDensities() {
    neighborSearcher()->forEachNearbyPoint(
        positions(), _kernelRadius,
        UpdateDensity(_kernelRadius, _mass, densities().data()));
}

float CudaSphSystemData2::targetDensity() const { return _targetDensity; }

void CudaSphSystemData2::setTargetDensity(float targetDensity) {
    _targetDensity = targetDensity;

    computeMass();
}

float CudaSphSystemData2::targetSpacing() const { return _targetSpacing; }

void CudaSphSystemData2::setTargetSpacing(float spacing) {
    _targetSpacing = spacing;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

float CudaSphSystemData2::relativeKernelRadius() const {
    return _kernelRadiusOverTargetSpacing;
}

void CudaSphSystemData2::setRelativeKernelRadius(float relativeRadius) {
    _kernelRadiusOverTargetSpacing = relativeRadius;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

float CudaSphSystemData2::kernelRadius() const { return _kernelRadius; }

void CudaSphSystemData2::setKernelRadius(float kernelRadius) {
    _kernelRadius = kernelRadius;
    _targetSpacing = kernelRadius / _kernelRadiusOverTargetSpacing;

    computeMass();
}

float CudaSphSystemData2::mass() const { return _mass; }

void CudaSphSystemData2::buildNeighborSearcher() {
    CudaParticleSystemData2::buildNeighborSearcher(_kernelRadius);
}

void CudaSphSystemData2::buildNeighborListsAndUpdateDensities() {
    size_t n = numberOfParticles();

    _neighborStarts.resize(n);
    _neighborEnds.resize(n);

    auto neighborStarts = _neighborStarts.view();

    // Count nearby points
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(0) + numberOfParticles(),
        ForEachNeighborFunc<NoOpFunc, CountNearbyPointsFunc>(
            *_neighborSearcher, _kernelRadius, positions().data(), NoOpFunc(),
            CountNearbyPointsFunc(_neighborStarts.data())));

    // Make start/end point of neighbor list, and allocate neighbor list.
    thrust::inclusive_scan(_neighborStarts.begin(), _neighborStarts.end(),
                           _neighborEnds.begin());
    thrust::transform(_neighborEnds.begin(), _neighborEnds.end(),
                      _neighborStarts.begin(), _neighborStarts.begin(),
                      thrust::minus<uint32_t>());
    size_t rbeginIdx = _neighborEnds.size() > 0 ? _neighborEnds.size() - 1 : 0;
    uint32_t m = _neighborEnds[rbeginIdx];
    _neighborLists.resize(m, 0);

    // Build neighbor lists and update densities
    auto d = densities();
    thrust::fill(d.begin(), d.end(), 0.0f);
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(0) + numberOfParticles(),
        ForEachNeighborFunc<BuildNeighborListsAndUpdateDensitiesFunc, NoOpFunc>(
            *_neighborSearcher, _kernelRadius, positions().data(),
            BuildNeighborListsAndUpdateDensitiesFunc(
                _neighborStarts.data(), _neighborEnds.data(), _kernelRadius,
                _mass, _neighborLists.data(), d.data()),
            NoOpFunc()));
}

void CudaSphSystemData2::set(const CudaSphSystemData2& other) {
    CudaParticleSystemData2::set(other);

    _targetDensity = other._targetDensity;
    _targetSpacing = other._targetSpacing;
    _kernelRadiusOverTargetSpacing = other._kernelRadiusOverTargetSpacing;
    _kernelRadius = other._kernelRadius;
    _densityIdx = other._densityIdx;
    _pressureIdx = other._pressureIdx;
}

CudaSphSystemData2& CudaSphSystemData2::operator=(
    const CudaSphSystemData2& other) {
    set(other);
    return (*this);
}

void CudaSphSystemData2::computeMass() {
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

    _mass = static_cast<float>(_targetDensity / maxNumberDensity);
}
