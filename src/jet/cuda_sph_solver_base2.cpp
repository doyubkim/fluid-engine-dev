// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include <pch.h>

#include <jet/cuda_sph_solver_base2.h>

using namespace jet;
using namespace experimental;

static double kTimeStepLimitBySpeedFactor = 0.4;
static double kTimeStepLimitByForceFactor = 0.25;

CudaSphSolverBase2::CudaSphSolverBase2() {
    _sphSystemData = std::make_shared<CudaSphSystemData2>();
    _forcesIdx = _sphSystemData->addVectorData();
    _smoothedVelIdx = _sphSystemData->addVectorData();

    setIsUsingFixedSubTimeSteps(false);
}

CudaSphSolverBase2::~CudaSphSolverBase2() {}

float CudaSphSolverBase2::negativePressureScale() const {
    return _negativePressureScale;
}

void CudaSphSolverBase2::setNegativePressureScale(
    float newNegativePressureScale) {
    _negativePressureScale = newNegativePressureScale;
}

float CudaSphSolverBase2::viscosityCoefficient() const {
    return _viscosityCoefficient;
}

void CudaSphSolverBase2::setViscosityCoefficient(
    float newViscosityCoefficient) {
    _viscosityCoefficient = newViscosityCoefficient;
}

float CudaSphSolverBase2::pseudoViscosityCoefficient() const {
    return _pseudoViscosityCoefficient;
}

void CudaSphSolverBase2::setPseudoViscosityCoefficient(
    float newPseudoViscosityCoefficient) {
    _pseudoViscosityCoefficient = newPseudoViscosityCoefficient;
}

float CudaSphSolverBase2::speedOfSound() const { return _speedOfSound; }

void CudaSphSolverBase2::setSpeedOfSound(float newSpeedOfSound) {
    _speedOfSound = std::max(newSpeedOfSound, kEpsilonF);
}

float CudaSphSolverBase2::timeStepLimitScale() const {
    return _timeStepLimitScale;
}

void CudaSphSolverBase2::setTimeStepLimitScale(float newScale) {
    _timeStepLimitScale = std::max(newScale, 0.0f);
}

CudaParticleSystemData2* CudaSphSolverBase2::particleSystemData() {
    return _sphSystemData.get();
}

const CudaParticleSystemData2* CudaSphSolverBase2::particleSystemData() const {
    return _sphSystemData.get();
}

CudaSphSystemData2* CudaSphSolverBase2::sphSystemData() {
    return _sphSystemData.get();
}

const CudaSphSystemData2* CudaSphSolverBase2::sphSystemData() const {
    return _sphSystemData.get();
}

unsigned int CudaSphSolverBase2::numberOfSubTimeSteps(
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

CudaArrayView1<float2> CudaSphSolverBase2::forces() const {
    return _sphSystemData->vectorDataAt(_forcesIdx);
}

CudaArrayView1<float2> CudaSphSolverBase2::smoothedVelocities() const {
    return _sphSystemData->vectorDataAt(_smoothedVelIdx);
}

#endif  // JET_USE_CUDA
