// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include <pch.h>

#include <jet/cuda_sph_solver_base3.h>

using namespace jet;

static double kTimeStepLimitBySpeedFactor = 0.4;
static double kTimeStepLimitByForceFactor = 0.25;

CudaSphSolverBase3::CudaSphSolverBase3() {
    _sphSystemData = std::make_shared<CudaSphSystemData3>();
    _forcesIdx = _sphSystemData->addVectorData();
    _smoothedVelIdx = _sphSystemData->addVectorData();

    setIsUsingFixedSubTimeSteps(false);
}

CudaSphSolverBase3::~CudaSphSolverBase3() {}

float CudaSphSolverBase3::negativePressureScale() const {
    return _negativePressureScale;
}

void CudaSphSolverBase3::setNegativePressureScale(
    float newNegativePressureScale) {
    _negativePressureScale = newNegativePressureScale;
}

float CudaSphSolverBase3::viscosityCoefficient() const {
    return _viscosityCoefficient;
}

void CudaSphSolverBase3::setViscosityCoefficient(
    float newViscosityCoefficient) {
    _viscosityCoefficient = newViscosityCoefficient;
}

float CudaSphSolverBase3::pseudoViscosityCoefficient() const {
    return _pseudoViscosityCoefficient;
}

void CudaSphSolverBase3::setPseudoViscosityCoefficient(
    float newPseudoViscosityCoefficient) {
    _pseudoViscosityCoefficient = newPseudoViscosityCoefficient;
}

float CudaSphSolverBase3::speedOfSound() const { return _speedOfSound; }

void CudaSphSolverBase3::setSpeedOfSound(float newSpeedOfSound) {
    _speedOfSound = std::max(newSpeedOfSound, kEpsilonF);
}

float CudaSphSolverBase3::timeStepLimitScale() const {
    return _timeStepLimitScale;
}

void CudaSphSolverBase3::setTimeStepLimitScale(float newScale) {
    _timeStepLimitScale = std::max(newScale, 0.0f);
}

const BoundingBox3F& CudaSphSolverBase3::container() const {
    return _container;
}

void CudaSphSolverBase3::setContainer(const BoundingBox3F& cont) {
    _container = cont;
}

CudaParticleSystemData3* CudaSphSolverBase3::particleSystemData() {
    return _sphSystemData.get();
}

const CudaParticleSystemData3* CudaSphSolverBase3::particleSystemData() const {
    return _sphSystemData.get();
}

CudaSphSystemData3* CudaSphSolverBase3::sphSystemData() {
    return _sphSystemData.get();
}

const CudaSphSystemData3* CudaSphSolverBase3::sphSystemData() const {
    return _sphSystemData.get();
}

unsigned int CudaSphSolverBase3::numberOfSubTimeSteps(
    double timeIntervalInSeconds) const {
    auto particles = sphSystemData();
    // size_t numberOfParticles = particles->numberOfParticles();
    // auto f = particles->forces();

    const double kernelRadius = particles->kernelRadius();
    const double mass = particles->mass();

    double maxForceMagnitude = 0.0;

    // for (size_t i = 0; i < numberOfParticles; ++i) {
    //     maxForceMagnitude = std::max(maxForceMagnitude, f[i].length());
    // }
    maxForceMagnitude = kGravityD;

    double timeStepLimitBySpeed =
        kTimeStepLimitBySpeedFactor * kernelRadius / _speedOfSound;
    double timeStepLimitByForce =
        kTimeStepLimitByForceFactor *
        std::sqrt(kernelRadius * mass / maxForceMagnitude);

    double desiredTimeStep =
        timeStepLimitScale() *
        std::min(timeStepLimitBySpeed, timeStepLimitByForce);

    return static_cast<unsigned int>(
        std::ceil(timeIntervalInSeconds / desiredTimeStep));
}

CudaArrayView1<float4> CudaSphSolverBase3::forces() const {
    return _sphSystemData->vectorDataAt(_forcesIdx);
}

CudaArrayView1<float4> CudaSphSolverBase3::smoothedVelocities() const {
    return _sphSystemData->vectorDataAt(_smoothedVelIdx);
}

#endif  // JET_USE_CUDA
