// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_CUDA

#include <jet/cuda_wc_sph_solver3.h>

using namespace jet;
using namespace experimental;

static double kTimeStepLimitBySpeedFactor = 0.4;
static double kTimeStepLimitByForceFactor = 0.25;

CudaWcSphSolver3::CudaWcSphSolver3()
    : CudaWcSphSolver3(kWaterDensityF, 0.1f, 1.8f) {}

CudaWcSphSolver3::CudaWcSphSolver3(float targetDensity, float targetSpacing,
                                   float relativeKernelRadius)
    : _targetDensity(targetDensity),
      _targetSpacing(targetSpacing),
      _relativeKernelRadius(relativeKernelRadius)
    : CudaSphSolverBase3() {}

CudaWcSphSolver3::~CudaWcSphSolver3() {}

float CudaWcSphSolver3::eosExponent() const { return _eosExponent; }

void CudaWcSphSolver3::setEosExponent(float newEosExponent) {
    _eosExponent = std::max(newEosExponent, 1.0f);
}

float CudaWcSphSolver3::speedOfSound() const { return _speedOfSound; }

void CudaWcSphSolver3::setSpeedOfSound(float newSpeedOfSound) {
    _speedOfSound = std::max(newSpeedOfSound, kEpsilonF);
}

float CudaWcSphSolver3::timeStepLimitScale() const {
    return _timeStepLimitScale;
}

void CudaWcSphSolver3::setTimeStepLimitScale(float newScale) {
    _timeStepLimitScale = std::max(newScale, 0.0f);
}

unsigned int CudaWcSphSolver3::numberOfSubTimeSteps(
    double timeIntervalInSeconds) const {
    auto particles = particleSystemData();
    size_t numberOfParticles = particles->numberOfParticles();
    // auto f = particles->forces();

    const double kernelRadius = particles->kernelRadius();
    const double mass = particles->mass();

    double maxForceMagnitude = 0.0;

    // for (size_t i = 0; i < numberOfParticles; ++i) {
    //     maxForceMagnitude = std::max(maxForceMagnitude, f[i].length());
    // }
    maxForceMagnitude = kGravity;

    double timeStepLimitBySpeed =
        kTimeStepLimitBySpeedFactor * kernelRadius / _speedOfSound;
    double timeStepLimitByForce =
        kTimeStepLimitByForceFactor *
        std::sqrt(kernelRadius * mass / maxForceMagnitude);

    double desiredTimeStep =
        _timeStepLimitScale *
        std::min(timeStepLimitBySpeed, timeStepLimitByForce);

    return static_cast<unsigned int>(
        std::ceil(timeIntervalInSeconds / desiredTimeStep));
}

CudaWcSphSolver3::Builder CudaWcSphSolver3::builder() { return Builder(); }

//

CudaWcSphSolver3 CudaWcSphSolver3::Builder::build() const {
    return CudaWcSphSolver3(_targetDensity, _targetSpacing,
                            _relativeKernelRadius);
}

CudaWcSphSolver3Ptr CudaWcSphSolver3::Builder::makeShared() const {
    return std::shared_ptr<CudaWcSphSolver3>(
        new CudaWcSphSolver3(_targetDensity, _targetSpacing,
                             _relativeKernelRadius),
        [](CudaWcSphSolver3* obj) { delete obj; });
}

#endif  // JET_USE_CUDA
