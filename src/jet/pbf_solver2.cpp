// Copyright (c) 2017 Doyub Kim

#include <pch.h>
#include <physics_helpers.h>
#include <jet/array_utils.h>
#include <jet/parallel.h>
#include <jet/sph_kernels2.h>
#include <jet/pbf_solver2.h>
#include <jet/timer.h>

#include <algorithm>

using namespace jet;

PbfSolver2::PbfSolver2() {
    setParticleSystemData(std::make_shared<SphSystemData2>());
    setIsUsingFixedSubTimeSteps(true);
    setNumberOfFixedSubTimeSteps(10);
}

PbfSolver2::PbfSolver2(
    double targetDensity,
    double targetSpacing,
    double relativeKernelRadius) {
    auto sphParticles = std::make_shared<SphSystemData2>();
    setParticleSystemData(sphParticles);
    sphParticles->setTargetDensity(targetDensity);
    sphParticles->setTargetSpacing(targetSpacing);
    sphParticles->setRelativeKernelRadius(relativeKernelRadius);
    setIsUsingFixedSubTimeSteps(true);
    setNumberOfFixedSubTimeSteps(5);
}

PbfSolver2::~PbfSolver2() {
}

double PbfSolver2::pseudoViscosityCoefficient() const {
    return _pseudoViscosityCoefficient;
}

void PbfSolver2::setPseudoViscosityCoefficient(
    double newPseudoViscosityCoefficient) {
    _pseudoViscosityCoefficient
        = clamp(newPseudoViscosityCoefficient, 0.0, 1.0);
}

unsigned int PbfSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

void PbfSolver2::setMaxNumberOfIterations(unsigned int n) {
    _maxNumberOfIterations = n;
}

double PbfSolver2::lambdaRelaxation() const {
    return _lambdaRelaxation;
}

void PbfSolver2::setLambdaRelaxation(double eps) {
    _lambdaRelaxation = eps;
}

double PbfSolver2::antiClusteringDenominatorFactor() const {
    return _antiClusteringDenom;
}

void PbfSolver2::setAntiClusteringDenominatorFactor(double factor) {
    _antiClusteringDenom = factor;
}

double PbfSolver2::antiClusteringStrength() const {
    return _antiClusteringStrength;
}

void PbfSolver2::setAntiClusteringStrength(double strength) {
    _antiClusteringStrength = strength;
}

double PbfSolver2::antiClusteringExponent() const {
    return _antiClusteringExp;
}

void PbfSolver2::setAntiClusteringExponent(double exponent) {
    _antiClusteringExp = exponent;
}

SphSystemData2Ptr PbfSolver2::sphSystemData() const {
    return std::dynamic_pointer_cast<SphSystemData2>(particleSystemData());
}

void PbfSolver2::onAdvanceTimeStep(double timeStepInSeconds) {
    // Clear forces
    auto particles = sphSystemData();
    auto forces = particles->forces();
    setRange1(forces.size(), Vector2D(), &forces);

    Timer timer;
    updateCollider(timeStepInSeconds);
    JET_INFO << "Update collider took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    updateEmitter(timeStepInSeconds);
    JET_INFO << "Update emitter took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    predictPosition(timeStepInSeconds);
    JET_INFO << "Position prediction took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    updatePosition(timeStepInSeconds);
    JET_INFO << "Position update took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    computePseudoViscosity(timeStepInSeconds);
    JET_INFO << "Computing pseudo-viscosity took "
             << timer.durationInSeconds() << " seconds";

    // Some stats
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

void PbfSolver2::predictPosition(double timeStepInSeconds) {
    accumulateForces(timeStepInSeconds);

    auto particles = sphSystemData();
    const size_t n = particles->numberOfParticles();
    auto forces = particles->forces();
    auto velocities = particles->velocities();
    auto positions = particles->positions();
    const double mass = particles->mass();

    _originalPositions.resize(n);

    parallelFor(kZeroSize, n, [&] (size_t i) {
        _originalPositions[i] = positions[i];
    });

    parallelFor(kZeroSize, n, [&] (size_t i) {
        // Integrate velocity first
        velocities[i] += timeStepInSeconds * forces[i] / mass;

        // Integrate position.
        positions[i] += timeStepInSeconds * velocities[i];
    });

    particles->buildNeighborSearcher();
    particles->buildNeighborLists();
    particles->updateDensities();
}

void PbfSolver2::updatePosition(double timeStepInSeconds) {
    auto particles = sphSystemData();
    const auto& neighborLists = particles->neighborLists();
    const size_t n = particles->numberOfParticles();
    const double targetDensity = particles->targetDensity();
    const double h = particles->kernelRadius();
    auto x = particles->positions();
    auto v = particles->velocities();
    auto d = particles->densities();

    const SphStdKernel2 stdKernel(h);
    const SphSpikyKernel2 kernel(h);

    ParticleSystemData2::ScalarData lambdas(n);

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        // Calculate lambda
        parallelFor(kZeroSize, n, [&] (size_t i) {
            const auto& neighbors = neighborLists[i];
            Vector2D origin = x[i];
            Vector2D sumGradCAtCenter;
            double sumGradC = 0.0;

            // Constraint
            const double c = d[i] / targetDensity - 1.0;;

            // Gradient from neighbors
            for (size_t j : neighbors) {
                Vector2D neighborPosition = x[j];
                double dist = origin.distanceTo(neighborPosition);
                if (dist > 0.0) {
                    Vector2D dir = (neighborPosition - origin) / dist;
                    Vector2D gradW = kernel.gradient(dist, dir);
                    sumGradCAtCenter += gradW;
                    sumGradC += gradW.dot(gradW);
                }
            }

            // Gradient at center
            sumGradC += sumGradCAtCenter.dot(sumGradCAtCenter);
            sumGradC /= targetDensity;

            lambdas[i] = -c / (sumGradC + _lambdaRelaxation);
        });

        // Update position
        const double sCorrWdq = stdKernel(_antiClusteringDenom * h);
        parallelFor(kZeroSize, n, [&] (size_t i) {
            const auto& neighbors = neighborLists[i];
            Vector2D origin = x[i];
            Vector2D sum;

            for (size_t j : neighbors) {
                const Vector2D& neighborPosition = x[j];
                const double dist = origin.distanceTo(neighborPosition);
                if (dist > 0.0) {
                    const double sCorr
                        = -_antiClusteringStrength
                          * pow(stdKernel(dist) / sCorrWdq, _antiClusteringExp);
                    Vector2D dir = (neighborPosition - origin) / dist;
                    Vector2D gradW = kernel.gradient(dist, dir);
                    sum += (lambdas[i] + lambdas[j] + sCorr) * gradW;
                }
            }

            x[i] += sum / targetDensity;
        });

        // Update velocity
        parallelFor(kZeroSize, n, [&] (size_t i) {
            v[i] = (x[i] - _originalPositions[i]) / timeStepInSeconds;
        });

        // Resolve collision
        resolveCollision();
    }
}

void PbfSolver2::computePseudoViscosity(double timeStepInSeconds) {
    auto particles = sphSystemData();
    size_t numberOfParticles = particles->numberOfParticles();
    auto x = particles->positions();
    auto v = particles->velocities();
    auto d = particles->densities();

    const double mass = particles->mass();
    const SphSpikyKernel2 kernel(particles->kernelRadius());

    Array1<Vector2D> smoothedVelocities(numberOfParticles);

    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
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

    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        v[i] = lerp(
            v[i], smoothedVelocities[i], _pseudoViscosityCoefficient);
    });
}

PbfSolver2::Builder PbfSolver2::builder() {
    return Builder();
}


PbfSolver2::Builder&
PbfSolver2::Builder::withTargetDensity(double targetDensity) {
    _targetDensity = targetDensity;
    return *this;
}

PbfSolver2::Builder&
PbfSolver2::Builder::withTargetSpacing(double targetSpacing) {
    _targetSpacing = targetSpacing;
    return *this;
}

PbfSolver2::Builder&
PbfSolver2::Builder::withRelativeKernelRadius(
    double relativeKernelRadius) {
    _relativeKernelRadius = relativeKernelRadius;
    return *this;
}

PbfSolver2 PbfSolver2::Builder::build() const {
    return PbfSolver2(
        _targetDensity,
        _targetSpacing,
        _relativeKernelRadius);
}

PbfSolver2Ptr PbfSolver2::Builder::makeShared() const {
    return std::shared_ptr<PbfSolver2>(
        new PbfSolver2(
            _targetDensity,
            _targetSpacing,
            _relativeKernelRadius),
        [] (PbfSolver2* obj) {
            delete obj;
    });
}
