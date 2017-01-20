// Copyright (c) 2017 Doyub Kim

#include <pch.h>
#include <physics_helpers.h>
#include <jet/array_utils.h>
#include <jet/parallel.h>
#include <jet/sph_kernels3.h>
#include <jet/pbf_solver3.h>
#include <jet/timer.h>

#include <algorithm>

using namespace jet;

PbfSolver3::PbfSolver3() {
    setParticleSystemData(std::make_shared<SphSystemData3>());
    setIsUsingFixedSubTimeSteps(true);
    setNumberOfFixedSubTimeSteps(10);
}

PbfSolver3::PbfSolver3(
    double targetDensity,
    double targetSpacing,
    double relativeKernelRadius) {
    auto sphParticles = std::make_shared<SphSystemData3>();
    setParticleSystemData(sphParticles);
    sphParticles->setTargetDensity(targetDensity);
    sphParticles->setTargetSpacing(targetSpacing);
    sphParticles->setRelativeKernelRadius(relativeKernelRadius);
    setIsUsingFixedSubTimeSteps(true);
    setNumberOfFixedSubTimeSteps(5);
}

PbfSolver3::~PbfSolver3() {
}

double PbfSolver3::pseudoViscosityCoefficient() const {
    return _pseudoViscosityCoefficient;
}

void PbfSolver3::setPseudoViscosityCoefficient(
    double newPseudoViscosityCoefficient) {
    _pseudoViscosityCoefficient
        = clamp(newPseudoViscosityCoefficient, 0.0, 1.0);
}

unsigned int PbfSolver3::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

void PbfSolver3::setMaxNumberOfIterations(unsigned int n) {
    _maxNumberOfIterations = n;
}

double PbfSolver3::lambdaRelaxation() const {
    return _lambdaRelaxation;
}

void PbfSolver3::setLambdaRelaxation(double eps) {
    _lambdaRelaxation = eps;
}

double PbfSolver3::antiClusteringDenominatorFactor() const {
    return _antiClusteringDenom;
}

void PbfSolver3::setAntiClusteringDenominatorFactor(double factor) {
    _antiClusteringDenom = factor;
}

double PbfSolver3::antiClusteringStrength() const {
    return _antiClusteringStrength;
}

void PbfSolver3::setAntiClusteringStrength(double strength) {
    _antiClusteringStrength = strength;
}

double PbfSolver3::antiClusteringExponent() const {
    return _antiClusteringExp;
}

void PbfSolver3::setAntiClusteringExponent(double exponent) {
    _antiClusteringExp = exponent;
}

void PbfSolver3::setVorticityConfinementStrength(double strength) {
    _vorticityConfinementStrength = strength;
}

double PbfSolver3::vorticityConfinementStrength() const {
    return _vorticityConfinementStrength;
}

SphSystemData3Ptr PbfSolver3::sphSystemData() const {
    return std::dynamic_pointer_cast<SphSystemData3>(particleSystemData());
}

void PbfSolver3::onAdvanceTimeStep(double timeStepInSeconds) {
    // Clear forces
    auto particles = sphSystemData();
    auto forces = particles->forces();
    setRange1(forces.size(), Vector3D(), &forces);

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

void PbfSolver3::predictPosition(double timeStepInSeconds) {
    auto particles = sphSystemData();

    accumulateForces(timeStepInSeconds);

    if (_vorticityConfinementStrength > 0.0) {
        particles->buildNeighborSearcher();
        particles->buildNeighborLists();
        computeVorticityConfinement();
    }

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

void PbfSolver3::updatePosition(double timeStepInSeconds) {
    auto particles = sphSystemData();
    const auto& neighborLists = particles->neighborLists();
    const size_t n = particles->numberOfParticles();
    const double targetDensity = particles->targetDensity();
    const double h = particles->kernelRadius();
    auto x = particles->positions();
    auto v = particles->velocities();
    auto d = particles->densities();

    const SphStdKernel3 stdKernel(h);
    const SphSpikyKernel3 kernel(h);

    ParticleSystemData3::ScalarData lambdas(n);

    for (unsigned int iter = 0; iter < _maxNumberOfIterations; ++iter) {
        // Calculate lambda
        parallelFor(kZeroSize, n, [&] (size_t i) {
            const auto& neighbors = neighborLists[i];
            Vector3D origin = x[i];
            Vector3D sumGradCAtCenter;
            double sumGradC = 0.0;

            // Constraint
            const double c = d[i] / targetDensity - 1.0;

            // Gradient from neighbors
            for (size_t j : neighbors) {
                Vector3D neighborPosition = x[j];
                double dist = origin.distanceTo(neighborPosition);
                if (dist > 0.0) {
                    Vector3D dir = (neighborPosition - origin) / dist;
                    Vector3D gradW = kernel.gradient(dist, dir);
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
            Vector3D origin = x[i];
            Vector3D sum;

            for (size_t j : neighbors) {
                const Vector3D& neighborPosition = x[j];
                const double dist = origin.distanceTo(neighborPosition);
                if (dist > 0.0) {
                    const double sCorr
                        = -_antiClusteringStrength
                          * pow(stdKernel(dist) / sCorrWdq, _antiClusteringExp);
                    Vector3D dir = (neighborPosition - origin) / dist;
                    Vector3D gradW = kernel.gradient(dist, dir);
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

void PbfSolver3::computePseudoViscosity(double timeStepInSeconds) {
    auto particles = sphSystemData();
    size_t numberOfParticles = particles->numberOfParticles();
    auto x = particles->positions();
    auto v = particles->velocities();
    auto d = particles->densities();

    const double mass = particles->mass();
    const SphSpikyKernel3 kernel(particles->kernelRadius());

    Array1<Vector3D> smoothedVelocities(numberOfParticles);

    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        double weightSum = 0.0;
        Vector3D smoothedVelocity;

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

void PbfSolver3::computeVorticityConfinement() {
    auto particles = sphSystemData();
    const size_t n = particles->numberOfParticles();
    const auto x = particles->positions();
    const auto v = particles->velocities();
    auto f = particles->forces();

    const double targetDensity = particles->targetDensity();
    const double mass = particles->mass();
    const SphSpikyKernel3 kernel(particles->kernelRadius());

    // Compute w
    ParticleSystemData3::VectorData w(n);
    parallelFor(kZeroSize, n, [&](size_t i) {
        const auto& neighbors = particles->neighborLists()[i];
        for (size_t j : neighbors) {
            const double dist = x[i].distanceTo(x[j]);
            if (dist > 0.0) {
                const Vector3D dir = (x[j] - x[i]) / dist;
                const Vector3D gradW = kernel.gradient(dist, dir);
                w[i] += (v[j] - v[i]).cross(gradW);
            }
        }
        w[i] *= mass / targetDensity;
    });

    // Compute force
    parallelFor(kZeroSize, n, [&](size_t i) {
        const auto& neighbors = particles->neighborLists()[i];
        Vector3D gradVor;
        for (size_t j : neighbors) {
            const double dist = x[i].distanceTo(x[j]);
            if (dist > 0.0) {
                const Vector3D dir = (x[j] - x[i]) / dist;
                const Vector3D gradW = kernel.gradient(dist, dir);
                gradVor += w[i].length() * gradW;
            }
        }
        gradVor *= mass / targetDensity;

        if (gradVor.lengthSquared() > 0.0) {
            const Vector3D n = gradVor.normalized();

            // f = e(N x w)
            f[i] += _vorticityConfinementStrength * n.cross(w[i]);
        }
    });
}

PbfSolver3::Builder PbfSolver3::builder() {
    return Builder();
}


PbfSolver3::Builder&
PbfSolver3::Builder::withTargetDensity(double targetDensity) {
    _targetDensity = targetDensity;
    return *this;
}

PbfSolver3::Builder&
PbfSolver3::Builder::withTargetSpacing(double targetSpacing) {
    _targetSpacing = targetSpacing;
    return *this;
}

PbfSolver3::Builder&
PbfSolver3::Builder::withRelativeKernelRadius(
    double relativeKernelRadius) {
    _relativeKernelRadius = relativeKernelRadius;
    return *this;
}

PbfSolver3 PbfSolver3::Builder::build() const {
    return PbfSolver3(
        _targetDensity,
        _targetSpacing,
        _relativeKernelRadius);
}

PbfSolver3Ptr PbfSolver3::Builder::makeShared() const {
    return std::shared_ptr<PbfSolver3>(
        new PbfSolver3(
            _targetDensity,
            _targetSpacing,
            _relativeKernelRadius),
        [] (PbfSolver3* obj) {
            delete obj;
    });
}
