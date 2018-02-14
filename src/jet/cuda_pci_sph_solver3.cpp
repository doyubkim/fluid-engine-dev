// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_CUDA

#include <jet/bcc_lattice_point_generator.h>
#include <jet/cuda_pci_sph_solver3.h>
#include <jet/sph_kernels3.h>

using namespace jet;
using namespace experimental;

// Heuristically chosen
const float kDefaultTimeStepLimitScale = 5.0f;

CudaPciSphSolver3::CudaPciSphSolver3() {
    setTimeStepLimitScale(kDefaultTimeStepLimitScale);
}

CudaPciSphSolver3::CudaPciSphSolver3(float targetDensity, float targetSpacing,
                                     float relativeKernelRadius)
    : CudaSphSolverBase3() {
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

CudaPciSphSolver3::~CudaPciSphSolver3() {}

float CudaPciSphSolver3::maxDensityErrorRatio() const {
    return _maxDensityErrorRatio;
}

void CudaPciSphSolver3::setMaxDensityErrorRatio(float ratio) {
    _maxDensityErrorRatio = std::max(ratio, 0.0f);
}

unsigned int CudaPciSphSolver3::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

void CudaPciSphSolver3::setMaxNumberOfIterations(unsigned int n) {
    _maxNumberOfIterations = n;
}

CudaArrayView1<float4> CudaPciSphSolver3::tempPositions() const {
    return sphSystemData()->vectorDataAt(_tempPositionsIdx);
}

CudaArrayView1<float4> CudaPciSphSolver3::tempVelocities() const {
    return sphSystemData()->vectorDataAt(_tempVelocitiesIdx);
}

CudaArrayView1<float> CudaPciSphSolver3::tempDensities() const {
    return sphSystemData()->floatDataAt(_tempDensitiesIdx);
}

CudaArrayView1<float4> CudaPciSphSolver3::pressureForces() const {
    return sphSystemData()->vectorDataAt(_pressureForcesIdx);
}

CudaArrayView1<float> CudaPciSphSolver3::densityErrors() const {
    return sphSystemData()->floatDataAt(_densityErrorsIdx);
}

float CudaPciSphSolver3::computeDelta(float timeStepInSeconds) {
    auto particles = sphSystemData();
    const double kernelRadius = particles->kernelRadius();

    Array1<Vector3D> points;
    BccLatticePointGenerator pointsGenerator;
    Vector3D origin;
    BoundingBox3D sampleBound(origin, origin);
    sampleBound.expand(1.5 * kernelRadius);

    pointsGenerator.generate(sampleBound, particles->targetSpacing(), &points);

    SphSpikyKernel3 kernel(kernelRadius);

    double denom = 0;
    Vector3D denom1;
    double denom2 = 0;

    for (size_t i = 0; i < points.size(); ++i) {
        const Vector3D& point = points[i];
        double distanceSquared = point.lengthSquared();

        if (distanceSquared < kernelRadius * kernelRadius) {
            double distance = std::sqrt(distanceSquared);
            Vector3D direction =
                (distance > 0.0) ? point / distance : Vector3D();

            // grad(Wij)
            Vector3D gradWij = kernel.gradient(distance, direction);
            denom1 += gradWij;
            denom2 += gradWij.dot(gradWij);
        }
    }

    denom += -denom1.dot(denom1) - denom2;

    float beta = computeBeta(timeStepInSeconds);

    return static_cast<float>((std::fabs(denom) > 0.0) ? -1 / (beta * denom)
                                                       : 0);
}

float CudaPciSphSolver3::computeBeta(float timeStepInSeconds) {
    auto particles = sphSystemData();
    return 2.0f * square(particles->mass() * timeStepInSeconds /
                         particles->targetDensity());
}

CudaPciSphSolver3::Builder CudaPciSphSolver3::builder() { return Builder(); }

CudaPciSphSolver3 CudaPciSphSolver3::Builder::build() const {
    return CudaPciSphSolver3(_targetDensity, _targetSpacing,
                             _relativeKernelRadius);
}

CudaPciSphSolver3Ptr CudaPciSphSolver3::Builder::makeShared() const {
    return std::shared_ptr<CudaPciSphSolver3>(
        new CudaPciSphSolver3(_targetDensity, _targetSpacing,
                              _relativeKernelRadius),
        [](CudaPciSphSolver3* obj) { delete obj; });
}

#endif  // JET_USE_CUDA
