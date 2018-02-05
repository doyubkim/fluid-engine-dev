// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/bcc_lattice_point_generator.h>
#include <jet/parallel.h>
#include <jet/pci_sph_solver3.h>
#include <jet/sph_kernels3.h>

#include <algorithm>

using namespace jet;

// Heuristically chosen
const double kDefaultTimeStepLimitScale = 5.0;

PciSphSolver3::PciSphSolver3() {
    setTimeStepLimitScale(kDefaultTimeStepLimitScale);
}

PciSphSolver3::PciSphSolver3(
    double targetDensity,
    double targetSpacing,
    double relativeKernelRadius)
: SphSolver3(targetDensity, targetSpacing, relativeKernelRadius) {
    setTimeStepLimitScale(kDefaultTimeStepLimitScale);
}

PciSphSolver3::~PciSphSolver3() {
}

double PciSphSolver3::maxDensityErrorRatio() const {
    return _maxDensityErrorRatio;
}

void PciSphSolver3::setMaxDensityErrorRatio(double ratio) {
    _maxDensityErrorRatio = std::max(ratio, 0.0);
}

unsigned int PciSphSolver3::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

void PciSphSolver3::setMaxNumberOfIterations(unsigned int n) {
    _maxNumberOfIterations = n;
}

void PciSphSolver3::accumulatePressureForce(
    double timeIntervalInSeconds) {
    auto particles = sphSystemData();
    const size_t numberOfParticles = particles->numberOfParticles();
    const double delta = computeDelta(timeIntervalInSeconds);
    const double targetDensity = particles->targetDensity();
    const double mass = particles->mass();

    auto p = particles->pressures();
    auto d = particles->densities();
    auto x = particles->positions();
    auto v = particles->velocities();
    auto f = particles->forces();

    // Predicted density ds
    Array1<double> ds(numberOfParticles, 0.0);

    SphStdKernel3 kernel(particles->kernelRadius());

    // Initialize buffers
    parallelFor(
        kZeroSize,
        numberOfParticles,
        [&] (size_t i) {
            p[i] = 0.0;
            _pressureForces[i] = Vector3D();
            _densityErrors[i] = 0.0;
            ds[i] = d[i];
        });

    unsigned int maxNumIter = 0;
    double maxDensityError;
    double densityErrorRatio = 0.0;

    for (unsigned int k = 0; k < _maxNumberOfIterations; ++k) {
        // Predict velocity and position
        parallelFor(
            kZeroSize,
            numberOfParticles,
            [&] (size_t i) {
                _tempVelocities[i]
                    = v[i]
                    + timeIntervalInSeconds / mass
                    * (f[i] + _pressureForces[i]);
                _tempPositions[i]
                    = x[i] + timeIntervalInSeconds * _tempVelocities[i];
            });

        // Resolve collisions
        resolveCollision(
            _tempPositions,
            _tempVelocities);

        // Compute pressure from density error
        parallelFor(
            kZeroSize,
            numberOfParticles,
            [&] (size_t i) {
                double weightSum = 0.0;
                const auto& neighbors = particles->neighborLists()[i];

                for (size_t j : neighbors) {
                    double dist
                        = _tempPositions[j].distanceTo(_tempPositions[i]);
                    weightSum += kernel(dist);
                }
                weightSum += kernel(0);

                double density = mass * weightSum;
                double densityError = (density - targetDensity);
                double pressure = delta * densityError;

                if (pressure < 0.0) {
                    pressure *= negativePressureScale();
                    densityError *= negativePressureScale();
                }

                p[i] += pressure;
                ds[i] = density;
                _densityErrors[i] = densityError;
            });

        // Compute pressure gradient force
        _pressureForces.set(Vector3D());
        SphSolver3::accumulatePressureForce(
            x, ds.constAccessor(), p, _pressureForces.accessor());

        // Compute max density error
        maxDensityError = 0.0;
        for (size_t i = 0; i < numberOfParticles; ++i) {
            maxDensityError = absmax(maxDensityError, _densityErrors[i]);
        }

        densityErrorRatio = maxDensityError / targetDensity;
        maxNumIter = k + 1;

        if (std::fabs(densityErrorRatio) < _maxDensityErrorRatio) {
            break;
        }
    }

    JET_INFO << "Number of PCI iterations: " << maxNumIter;
    JET_INFO << "Max density error after PCI iteration: " << maxDensityError;
    if (std::fabs(densityErrorRatio) > _maxDensityErrorRatio) {
        JET_WARN << "Max density error ratio is greater than the threshold!";
        JET_WARN << "Ratio: " << densityErrorRatio
                 << " Threshold: " << _maxDensityErrorRatio;
    }

    // Accumulate pressure force
    parallelFor(
        kZeroSize,
        numberOfParticles,
        [this, &f](size_t i) {
            f[i] += _pressureForces[i];
        });
}

void PciSphSolver3::onBeginAdvanceTimeStep(double timeStepInSeconds) {
    SphSolver3::onBeginAdvanceTimeStep(timeStepInSeconds);

    // Allocate temp buffers
    size_t numberOfParticles = particleSystemData()->numberOfParticles();
    _tempPositions.resize(numberOfParticles);
    _tempVelocities.resize(numberOfParticles);
    _pressureForces.resize(numberOfParticles);
    _densityErrors.resize(numberOfParticles);
}

double PciSphSolver3::computeDelta(double timeStepInSeconds) {
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

    return (std::fabs(denom) > 0.0) ?
        -1 / (computeBeta(timeStepInSeconds) * denom) : 0;
}

double PciSphSolver3::computeBeta(double timeStepInSeconds) {
    auto particles = sphSystemData();
    return 2.0 * square(particles->mass() * timeStepInSeconds
        / particles->targetDensity());
}

PciSphSolver3::Builder PciSphSolver3::builder() {
    return Builder();
}

PciSphSolver3 PciSphSolver3::Builder::build() const {
    return PciSphSolver3(
        _targetDensity,
        _targetSpacing,
        _relativeKernelRadius);
}

PciSphSolver3Ptr PciSphSolver3::Builder::makeShared() const {
    return std::shared_ptr<PciSphSolver3>(
        new PciSphSolver3(
            _targetDensity,
            _targetSpacing,
            _relativeKernelRadius),
        [] (PciSphSolver3* obj) {
            delete obj;
        });
}
