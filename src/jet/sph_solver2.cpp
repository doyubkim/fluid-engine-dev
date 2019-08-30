// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <physics_helpers.h>
#include <jet/parallel.h>
#include <jet/sph_kernels2.h>
#include <jet/sph_solver2.h>
#include <jet/timer.h>

#include <algorithm>

using namespace jet;

static double kTimeStepLimitBySpeedFactor = 0.4;
static double kTimeStepLimitByForceFactor = 0.25;

SphSolver2::SphSolver2() {
    setParticleSystemData(std::make_shared<SphSystemData2>());
    setIsUsingFixedSubTimeSteps(false);
}

SphSolver2::SphSolver2(
    double targetDensity,
    double targetSpacing,
    double relativeKernelRadius) {
    auto sphParticles = std::make_shared<SphSystemData2>();
    setParticleSystemData(sphParticles);
    sphParticles->setTargetDensity(targetDensity);
    sphParticles->setTargetSpacing(targetSpacing);
    sphParticles->setRelativeKernelRadius(relativeKernelRadius);
    setIsUsingFixedSubTimeSteps(false);
}

SphSolver2::~SphSolver2() {
}

double SphSolver2::eosExponent() const {
    return _eosExponent;
}

void SphSolver2::setEosExponent(double newEosExponent) {
    _eosExponent = std::max(newEosExponent, 1.0);
}

double SphSolver2::negativePressureScale() const {
    return _negativePressureScale;
}

void SphSolver2::setNegativePressureScale(
    double newNegativePressureScale) {
    _negativePressureScale = clamp(newNegativePressureScale, 0.0, 1.0);
}

double SphSolver2::viscosityCoefficient() const {
    return _viscosityCoefficient;
}

void SphSolver2::setViscosityCoefficient(double newViscosityCoefficient) {
    _viscosityCoefficient = std::max(newViscosityCoefficient, 0.0);
}

double SphSolver2::pseudoViscosityCoefficient() const {
    return _pseudoViscosityCoefficient;
}

void SphSolver2::setPseudoViscosityCoefficient(
    double newPseudoViscosityCoefficient) {
    _pseudoViscosityCoefficient
        = std::max(newPseudoViscosityCoefficient, 0.0);
}

double SphSolver2::speedOfSound() const {
    return _speedOfSound;
}

void SphSolver2::setSpeedOfSound(double newSpeedOfSound) {
    _speedOfSound = std::max(newSpeedOfSound, kEpsilonD);
}

double SphSolver2::timeStepLimitScale() const {
    return _timeStepLimitScale;
}

void SphSolver2::setTimeStepLimitScale(double newScale) {
    _timeStepLimitScale = std::max(newScale, 0.0);
}

SphSystemData2Ptr SphSolver2::sphSystemData() const {
    return std::dynamic_pointer_cast<SphSystemData2>(particleSystemData());
}

unsigned int SphSolver2::numberOfSubTimeSteps(
    double timeIntervalInSeconds) const {
    auto particles = sphSystemData();
    size_t numberOfParticles = particles->numberOfParticles();
    auto f = particles->forces();

    const double kernelRadius = particles->kernelRadius();
    const double mass = particles->mass();

    double maxForceMagnitude = 0.0;

    for (size_t i = 0; i < numberOfParticles; ++i) {
        maxForceMagnitude = std::max(maxForceMagnitude, f[i].length());
    }

    double timeStepLimitBySpeed
        = kTimeStepLimitBySpeedFactor * kernelRadius / _speedOfSound;
    double timeStepLimitByForce
        = kTimeStepLimitByForceFactor
        * std::sqrt(kernelRadius * mass / maxForceMagnitude);

    double desiredTimeStep
        = _timeStepLimitScale
        * std::min(timeStepLimitBySpeed, timeStepLimitByForce);

    return static_cast<unsigned int>(
        std::ceil(timeIntervalInSeconds / desiredTimeStep));
}

void SphSolver2::accumulateForces(double timeStepInSeconds) {
    accumulateNonPressureForces(timeStepInSeconds);
    accumulatePressureForce(timeStepInSeconds);
}

void SphSolver2::onBeginAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);

    auto particles = sphSystemData();

    Timer timer;
    particles->buildNeighborSearcher();
    particles->buildNeighborLists();
    particles->updateDensities();

    JET_INFO << "Building neighbor lists and updating densities took "
             << timer.durationInSeconds()
             << " seconds";
}

void SphSolver2::onEndAdvanceTimeStep(double timeStepInSeconds) {
    computePseudoViscosity(timeStepInSeconds);

    auto particles = sphSystemData();
    size_t numberOfParticles = particles->numberOfParticles();
    auto densities = particles->densities();

    double maxDensity = 0.0;
    for (size_t i = 0; i < numberOfParticles; ++i) {
        maxDensity = std::max(maxDensity, densities[i]);
    }

    JET_INFO << "Max density: " << maxDensity << " "
             << "Max density / target density ratio: "
             << maxDensity / particles->targetDensity();
}

void SphSolver2::accumulateNonPressureForces(double timeStepInSeconds) {
    ParticleSystemSolver2::accumulateForces(timeStepInSeconds);
    accumulateViscosityForce();
}

void SphSolver2::accumulatePressureForce(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);

    auto particles = sphSystemData();
    auto x = particles->positions();
    auto d = particles->densities();
    auto p = particles->pressures();
    auto f = particles->forces();

    computePressure();
    accumulatePressureForce(x, d, p, f);
}

void SphSolver2::computePressure() {
    auto particles = sphSystemData();
    size_t numberOfParticles = particles->numberOfParticles();
    auto d = particles->densities();
    auto p = particles->pressures();

    // See Murnaghan-Tait equation of state from
    // https://en.wikipedia.org/wiki/Tait_equation
    const double targetDensity = particles->targetDensity();
    const double eosScale = targetDensity * square(_speedOfSound);

    parallelFor(
        kZeroSize,
        numberOfParticles,
        [&](size_t i) {
            p[i] = computePressureFromEos(
                d[i],
                targetDensity,
                eosScale,
                eosExponent(),
                negativePressureScale());
        });
}

void SphSolver2::accumulatePressureForce(
    const ConstArrayAccessor1<Vector2D>& positions,
    const ConstArrayAccessor1<double>& densities,
    const ConstArrayAccessor1<double>& pressures,
    ArrayAccessor1<Vector2D> pressureForces) {
    auto particles = sphSystemData();
    size_t numberOfParticles = particles->numberOfParticles();

    const double massSquared = square(particles->mass());
    const SphSpikyKernel2 kernel(particles->kernelRadius());

    parallelFor(
        kZeroSize,
        numberOfParticles,
        [&](size_t i) {
            const auto& neighbors = particles->neighborLists()[i];
            for (size_t j : neighbors) {
                double dist = positions[i].distanceTo(positions[j]);

                if (dist > 0.0) {
                    Vector2D dir = (positions[j] - positions[i]) / dist;
                    pressureForces[i] -= massSquared
                        * (pressures[i] / (densities[i] * densities[i])
                            + pressures[j] / (densities[j] * densities[j]))
                        * kernel.gradient(dist, dir);
                }
            }
        });
}


void SphSolver2::accumulateViscosityForce() {
    auto particles = sphSystemData();
    size_t numberOfParticles = particles->numberOfParticles();
    auto x = particles->positions();
    auto v = particles->velocities();
    auto d = particles->densities();
    auto f = particles->forces();

    const double massSquared = square(particles->mass());
    const SphSpikyKernel2 kernel(particles->kernelRadius());

    parallelFor(
        kZeroSize,
        numberOfParticles,
        [&](size_t i) {
            const auto& neighbors = particles->neighborLists()[i];
            for (size_t j : neighbors) {
                double dist = x[i].distanceTo(x[j]);

                f[i] += viscosityCoefficient() * massSquared
                    * (v[j] - v[i]) / d[j]
                    * kernel.secondDerivative(dist);
            }
        });
}

void SphSolver2::computePseudoViscosity(double timeStepInSeconds) {
    auto particles = sphSystemData();
    size_t numberOfParticles = particles->numberOfParticles();
    auto x = particles->positions();
    auto v = particles->velocities();
    auto d = particles->densities();

    const double mass = particles->mass();
    const SphSpikyKernel2 kernel(particles->kernelRadius());

    Array1<Vector2D> smoothedVelocities(numberOfParticles);

    parallelFor(
        kZeroSize,
        numberOfParticles,
        [&](size_t i) {
            double weightSum = 0.0;
            Vector2D smoothedVelocity;

            const auto& neighbors = particles->neighborLists()[i];
            for (size_t j : neighbors) {
                double dist = x[i].distanceTo(x[j]);
                double wj = mass / d[j] * kernel(dist);
                weightSum += wj;
                smoothedVelocity += wj * v[j];
            }

            double wi = mass / d[i];
            weightSum += wi;
            smoothedVelocity += wi * v[i];

            if (weightSum > 0.0) {
                smoothedVelocity /= weightSum;
            }

            smoothedVelocities[i] = smoothedVelocity;
        });

    double factor = timeStepInSeconds * _pseudoViscosityCoefficient;
    factor = clamp(factor, 0.0, 1.0);

    parallelFor(
        kZeroSize,
        numberOfParticles,
        [&](size_t i) {
            v[i] = lerp(
                v[i], smoothedVelocities[i], factor);
        });
}

SphSolver2::Builder SphSolver2::builder() {
    return Builder();
}

SphSolver2 SphSolver2::Builder::build() const {
    return SphSolver2(
        _targetDensity,
        _targetSpacing,
        _relativeKernelRadius);
}

SphSolver2Ptr SphSolver2::Builder::makeShared() const {
    return std::shared_ptr<SphSolver2>(
        new SphSolver2(
            _targetDensity,
            _targetSpacing,
            _relativeKernelRadius),
        [] (SphSolver2* obj) {
            delete obj;
    });
}
