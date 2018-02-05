// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/triangle_point_generator.h>
#include <jet/parallel.h>
#include <jet/pci_sph_solver2.h>
#include <jet/sph_kernels2.h>

#include <algorithm>

using namespace jet;

// Heuristically chosen
const double kDefaultTimeStepLimitScale = 5.0;

PciSphSolver2::PciSphSolver2() {
    setTimeStepLimitScale(kDefaultTimeStepLimitScale);
}

PciSphSolver2::PciSphSolver2(
    double targetDensity,
    double targetSpacing,
    double relativeKernelRadius)
: SphSolver2(targetDensity, targetSpacing, relativeKernelRadius) {
    setTimeStepLimitScale(kDefaultTimeStepLimitScale);
}

PciSphSolver2::~PciSphSolver2() {
}

double PciSphSolver2::maxDensityErrorRatio() const {
    return _maxDensityErrorRatio;
}

void PciSphSolver2::setMaxDensityErrorRatio(double ratio) {
    _maxDensityErrorRatio = std::max(ratio, 0.0);
}

unsigned int PciSphSolver2::maxNumberOfIterations() const {
    return _maxNumberOfIterations;
}

void PciSphSolver2::setMaxNumberOfIterations(unsigned int n) {
    _maxNumberOfIterations = n;
}

void PciSphSolver2::accumulatePressureForce(
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

    SphStdKernel2 kernel(particles->kernelRadius());

    // Initialize buffers
    parallelFor(
        kZeroSize,
        numberOfParticles,
        [&] (size_t i) {
            p[i] = 0.0;
            _pressureForces[i] = Vector2D();
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
        _pressureForces.set(Vector2D());
        SphSolver2::accumulatePressureForce(
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

void PciSphSolver2::onBeginAdvanceTimeStep(double timeStepInSeconds) {
    SphSolver2::onBeginAdvanceTimeStep(timeStepInSeconds);

    // Allocate temp buffers
    size_t numberOfParticles = particleSystemData()->numberOfParticles();
    _tempPositions.resize(numberOfParticles);
    _tempVelocities.resize(numberOfParticles);
    _pressureForces.resize(numberOfParticles);
    _densityErrors.resize(numberOfParticles);
}

double PciSphSolver2::computeDelta(double timeStepInSeconds) {
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

    return (std::fabs(denom) > 0.0) ?
        -1 / (computeBeta(timeStepInSeconds) * denom) : 0;
}

double PciSphSolver2::computeBeta(double timeStepInSeconds) {
    auto particles = sphSystemData();
    return 2.0 * square(particles->mass() * timeStepInSeconds
        / particles->targetDensity());
}

PciSphSolver2::Builder PciSphSolver2::builder() {
    return Builder();
}

PciSphSolver2 PciSphSolver2::Builder::build() const {
    return PciSphSolver2(
        _targetDensity,
        _targetSpacing,
        _relativeKernelRadius);
}

PciSphSolver2Ptr PciSphSolver2::Builder::makeShared() const {
    return std::shared_ptr<PciSphSolver2>(
        new PciSphSolver2(
            _targetDensity,
            _targetSpacing,
            _relativeKernelRadius),
        [] (PciSphSolver2* obj) {
            delete obj;
        });
}
