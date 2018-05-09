// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_CUDA

#include <jet/cuda_pci_sph_solver2.h>
#include <jet/sph_kernels2.h>
#include <jet/triangle_point_generator.h>

using namespace jet;
using namespace experimental;

// Heuristically chosen
const float kDefaultTimeStepLimitScale = 5.0f;

CudaPciSphSolver2::CudaPciSphSolver2() {
    setTimeStepLimitScale(kDefaultTimeStepLimitScale);
}

CudaPciSphSolver2::CudaPciSphSolver2(float targetDensity, float targetSpacing,
                                     float relativeKernelRadius)
    : CudaSphSolverBase2() {
    auto sph = sphSystemData();
    sph->setTargetDensity(targetDensity);
    sph->setTargetSpacing(targetSpacing);
    sph->setRelativeKernelRadius(relativeKernelRadius);

    _tempPositionsIdx = sph->addVectorData();
    _tempVelocitiesIdx = sph->addVectorData();
    _tempDensitiesIdx = sph->addFloatData();
    _pressureForcesIdx = sph->addVectorData();
    _densityErrorsIdx = sph->addFloatData();

    setTimeStepLimitScale(kDefaultTimeStepLimitScale);
}

CudaPciSphSolver2::~CudaPciSphSolver2() {}

float CudaPciSphSolver2::maxDensityErrorRatio() const {
    return _maxDensityErrorRatio;
}

void CudaPciSphSolver2::setMaxDensityErrorRatio(float ratio) {
    _maxDensityErrorRatio = std::max(ratio, 0.0f);
}

unsigned int CudaPciSphSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

void CudaPciSphSolver2::setMaxNumberOfIterations(unsigned int n) {
    _maxNumberOfIterations = n;
}

CudaArrayView1<float2> CudaPciSphSolver2::tempPositions() {
    return sphSystemData()->vectorDataAt(_tempPositionsIdx);
}

CudaArrayView1<float2> CudaPciSphSolver2::tempVelocities() {
    return sphSystemData()->vectorDataAt(_tempVelocitiesIdx);
}

CudaArrayView1<float> CudaPciSphSolver2::tempDensities() {
    return sphSystemData()->floatDataAt(_tempDensitiesIdx);
}

CudaArrayView1<float2> CudaPciSphSolver2::pressureForces() {
    return sphSystemData()->vectorDataAt(_pressureForcesIdx);
}

CudaArrayView1<float> CudaPciSphSolver2::densityErrors() {
    return sphSystemData()->floatDataAt(_densityErrorsIdx);
}

float CudaPciSphSolver2::computeDelta(float timeStepInSeconds) {
    auto particles = sphSystemData();
    const double kernelRadius = particles->kernelRadius();

    Array1<Vector2D> points;
    TrianglePointGenerator pointsGenerator;
    Vector2D origin;
    BoundingBox2D sampleBound(origin, origin);
    sampleBound.expand(1.5 * kernelRadius);

    pointsGenerator.generate(sampleBound, particles->targetSpacing(), &points);

    SphSpikyKernel2 kernel(kernelRadius);

    double denom = 0;
    Vector2D denom1;
    double denom2 = 0;

    for (size_t i = 0; i < points.size(); ++i) {
        const Vector2D& point = points[i];
        double distanceSquared = point.lengthSquared();

        if (distanceSquared < kernelRadius * kernelRadius) {
            double distance = std::sqrt(distanceSquared);
            Vector2D direction =
                (distance > 0.0) ? point / distance : Vector2D();

            // grad(Wij)
            Vector2D gradWij = kernel.gradient(distance, direction);
            denom1 += gradWij;
            denom2 += gradWij.dot(gradWij);
        }
    }

    denom += -denom1.dot(denom1) - denom2;

    float beta = computeBeta(timeStepInSeconds);

    return static_cast<float>((std::fabs(denom) > 0.0) ? -1 / (beta * denom)
                                                       : 0);
}

float CudaPciSphSolver2::computeBeta(float timeStepInSeconds) {
    auto particles = sphSystemData();
    return 2.0f * square(particles->mass() * timeStepInSeconds /
                         particles->targetDensity());
}

CudaPciSphSolver2::Builder CudaPciSphSolver2::builder() { return Builder(); }

CudaPciSphSolver2 CudaPciSphSolver2::Builder::build() const {
    return CudaPciSphSolver2(_targetDensity, _targetSpacing,
                             _relativeKernelRadius);
}

CudaPciSphSolver2Ptr CudaPciSphSolver2::Builder::makeShared() const {
    return std::shared_ptr<CudaPciSphSolver2>(
        new CudaPciSphSolver2(_targetDensity, _targetSpacing,
                              _relativeKernelRadius),
        [](CudaPciSphSolver2* obj) { delete obj; });
}

#endif  // JET_USE_CUDA
