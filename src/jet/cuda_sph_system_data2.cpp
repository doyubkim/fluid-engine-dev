// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_sph_system_data2.h>
#include <jet/sph_kernels2.h>
#include <jet/triangle_point_generator.h>

using namespace jet;

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

ConstCudaArrayView1<float> CudaSphSystemData2::densities() const {
    return floatDataAt(_densityIdx);
}

CudaArrayView1<float> CudaSphSystemData2::densities() {
    return floatDataAt(_densityIdx);
}

ConstCudaArrayView1<float> CudaSphSystemData2::pressures() const {
    return floatDataAt(_pressureIdx);
}

CudaArrayView1<float> CudaSphSystemData2::pressures() {
    return floatDataAt(_pressureIdx);
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

    for (size_t i = 0; i < points.length(); ++i) {
        const Vector2D& point = points[i];
        double sum = 0.0;

        for (size_t j = 0; j < points.length(); ++j) {
            const Vector2D& neighborPoint = points[j];
            sum += kernel(neighborPoint.distanceTo(point));
        }

        maxNumberDensity = std::max(maxNumberDensity, sum);
    }

    JET_ASSERT(maxNumberDensity > 0);

    _mass = static_cast<float>(_targetDensity / maxNumberDensity);
}
