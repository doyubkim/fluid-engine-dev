// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_CUDA

#include <jet/cuda_pci_sph_solver3.h>
#include <jet/cuda_sph_kernels3.h>

using namespace jet;
using namespace experimental;

namespace {

class InitializeBuffersAndComputeForces {
 public:
    inline InitializeBuffersAndComputeForces(
        float m, float h, float4 gravity, float viscosity,
        uint32_t* neighborStarts, uint32_t* neighborEnds,
        uint32_t* neighborLists, float4* positions, float4* velocities,
        float4* smoothedVelocities, float4* forces, float* densities,
        float* pressures, float4* pressureForces, float* densityErrors,
        float* densitiesPredicted)
        : _mass(m),
          _massSquared(m * m),
          _gravity(gravity),
          _viscosity(viscosity),
          _spikyKernel(h),
          _neighborStarts(neighborStarts),
          _neighborEnds(neighborEnds),
          _neighborLists(neighborLists),
          _positions(positions),
          _velocities(velocities),
          _smoothedVelocities(smoothedVelocities),
          _forces(forces),
          _densities(densities),
          _pressures(pressures),
          _pressureForces(pressureForces),
          _densitiesPredicted(densitiesPredicted),
          _densityErrors(densityErrors) {}

    template <typename Index>
    inline JET_CUDA_DEVICE void operator()(Index i) {
        // Initialize buffers
        _pressures[i] = 0.0f;
        _pressureForces[i] = make_float4(0, 0, 0, 0);
        _densityErrors[i] = 0.0f;
        _densitiesPredicted[i] = _densities[i];

        // Compute forces
        uint32_t ns = _neighborStarts[i];
        uint32_t ne = _neighborEnds[i];
        float4 x_i = _positions[i];
        float4 v_i = _velocities[i];
        float d_i = _densities[i];
        float4 f = _gravity;
        float w_i = _mass / d_i;
        float weightSum = w_i * _spikyKernel(0.0f);
        ;
        float4 smoothedVelocity = w_i * v_i;

        for (uint32_t jj = ns; jj < ne; ++jj) {
            uint32_t j = _neighborLists[jj];

            float4 r = _positions[j] - x_i;
            float dist = length(r);

            if (dist > 0.0f) {
                float4 dir = r / dist;

                float4 v_j = _velocities[j];
                float d_j = _densities[j];

                // Viscosity force
                f += _viscosity * _massSquared * (v_j - v_i) / d_j *
                     _spikyKernel.secondDerivative(dist);

                // Pseudo viscosity
                float w_j = _mass / d_j * _spikyKernel(dist);
                weightSum += w_j;
                smoothedVelocity += w_j * v_j;
            }
        }

        _forces[i] = f;

        smoothedVelocity /= weightSum;
        _smoothedVelocities[i] = smoothedVelocity;
    }

 private:
    float _mass;
    float _massSquared;
    float4 _gravity;
    float _viscosity;
    CudaSphSpikyKernel3 _spikyKernel;
    uint32_t* _neighborStarts;
    uint32_t* _neighborEnds;
    uint32_t* _neighborLists;
    float4* _positions;
    float4* _velocities;
    float4* _smoothedVelocities;
    float4* _forces;
    float* _densities;
    float* _pressures;
    float4* _pressureForces;
    float* _densitiesPredicted;
    float* _densityErrors;
};

#define LOWER_X 0.0f
#define UPPER_X 1.0f
#define LOWER_Y 0.0f
#define UPPER_Y 1.0f
#define LOWER_Z 0.0f
#define UPPER_Z 1.0f
#define BND_R -0.5f

class TimeIntegration {
 public:
    TimeIntegration(float dt, float smoothFactor, float4* positions,
                    float4* velocities, float4* newPositions,
                    float4* newVelocities, float4* smoothedVelocities,
                    float4* forces, float4* pressureForces)
        : _dt(dt),
          _smoothFactor(smoothFactor),
          _positions(positions),
          _velocities(velocities),
          _newPositions(newPositions),
          _newVelocities(newVelocities),
          _smoothedVelocities(smoothedVelocities),
          _forces(forces),
          _pressureForces(pressureForces) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index i) {
        float4 x = _positions[i];
        float4 v = _velocities[i];
        float4 s = _smoothedVelocities[i];
        float4 f = _forces[i];
        float4 pf = _pressureForces[i];

        v = (1.0f - _smoothFactor) * v + _smoothFactor * s;
        v += _dt * (f + pf);
        x += _dt * v;

        // TODO: Replace with collider
        if (x.x > UPPER_X) {
            x.x = UPPER_X;
            v.x *= BND_R;
        }
        if (x.x < LOWER_X) {
            x.x = LOWER_X;
            v.x *= BND_R;
        }
        if (x.y > UPPER_Y) {
            x.y = UPPER_Y;
            v.y *= BND_R;
        }
        if (x.y < LOWER_Y) {
            x.y = LOWER_Y;
            v.y *= BND_R;
        }
        if (x.z > UPPER_Z) {
            x.z = UPPER_Z;
            v.z *= BND_R;
        }
        if (x.z < LOWER_Z) {
            x.z = LOWER_Z;
            v.z *= BND_R;
        }

        _newPositions[i] = x;
        _newVelocities[i] = v;
    }

 private:
    float _dt;
    float _smoothFactor;
    float4* _positions;
    float4* _velocities;
    float4* _newPositions;
    float4* _newVelocities;
    float4* _smoothedVelocities;
    float4* _forces;
    float4* _pressureForces;
};

class ComputeDensityError {
 public:
    inline ComputeDensityError(float m, float h, float targetDensity,
                               float delta, float negativePressureScale,
                               uint32_t* neighborStarts, uint32_t* neighborEnds,
                               uint32_t* neighborLists, float4* positions,
                               float* densities, float* pressures,
                               float* densityErrors, float* densitiesPredicted)
        : _mass(m),
          _targetDensity(targetDensity),
          _delta(delta),
          _negativePressureScale(negativePressureScale),
          _neighborStarts(neighborStarts),
          _neighborEnds(neighborEnds),
          _neighborLists(neighborLists),
          _positions(positions),
          _densities(densities),
          _pressures(pressures),
          _densitiesPredicted(densitiesPredicted),
          _densityErrors(densityErrors),
          _stdKernel(h) {}

    template <typename Index>
    inline JET_CUDA_DEVICE void operator()(Index i) {
        uint32_t ns = _neighborStarts[i];
        uint32_t ne = _neighborEnds[i];
        float4 x_i = _positions[i];
        float kernelSum = _stdKernel(0.f);

        for (uint32_t jj = ns; jj < ne; ++jj) {
            uint32_t j = _neighborLists[jj];

            float4 r = _positions[j] - x_i;
            float dist = length(r);

            if (dist > 0.0f) {
                kernelSum += _stdKernel(dist);
            }
        }

        float density = _mass * kernelSum;
        float densityError = (density - _targetDensity);
        float pressure = _delta * densityError;

        if (pressure < 0.0f) {
            pressure *= _negativePressureScale;
            densityError *= _negativePressureScale;
        }

        _pressures[i] += pressure;
        _densitiesPredicted[i] = density;
        _densityErrors[i] = densityError;
    }

 private:
    float _mass;
    float _targetDensity;
    float _delta;
    float _negativePressureScale;
    uint32_t* _neighborStarts;
    uint32_t* _neighborEnds;
    uint32_t* _neighborLists;
    float4* _positions;
    float* _densities;
    float* _pressures;
    float* _densitiesPredicted;
    float* _densityErrors;
    CudaSphStdKernel3 _stdKernel;
};

class ComputePressureForces {
 public:
    inline ComputePressureForces(float m, float h, uint32_t* neighborStarts,
                                 uint32_t* neighborEnds,
                                 uint32_t* neighborLists, float4* positions,
                                 float4* pressureForces, float* densities,
                                 float* pressures)
        : _mass(m),
          _massSquared(m * m),
          _spikyKernel(h),
          _neighborStarts(neighborStarts),
          _neighborEnds(neighborEnds),
          _neighborLists(neighborLists),
          _positions(positions),
          _pressureForces(pressureForces),
          _densities(densities),
          _pressures(pressures) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index i) {
        uint32_t ns = _neighborStarts[i];
        uint32_t ne = _neighborEnds[i];

        float4 x_i = _positions[i];
        float d_i = _densities[i];
        float p_i = _pressures[i];

        float4 f = make_float4(0, 0, 0, 0);

        for (uint32_t jj = ns; jj < ne; ++jj) {
            uint32_t j = _neighborLists[jj];

            float4 r = _positions[j] - x_i;
            float dist = length(r);

            if (dist > 0.0f) {
                float4 dir = r / dist;

                float d_j = _densities[j];
                float p_j = _pressures[j];

                // Pressure force
                f -= _massSquared * (p_i / (d_i * d_i) + p_j / (d_j * d_j)) *
                     _spikyKernel.gradient(dist, dir);
            }
        }

        _pressureForces[i] = f;
    }

 private:
    float _mass;
    float _massSquared;
    CudaSphSpikyKernel3 _spikyKernel;
    uint32_t* _neighborStarts;
    uint32_t* _neighborEnds;
    uint32_t* _neighborLists;
    float4* _positions;
    float4* _pressureForces;
    float* _densities;
    float* _pressures;
};

}  // namespace

void CudaPciSphSolver3::onAdvanceTimeStep(double timeStepInSeconds) {
    auto sph = sphSystemData();

    float dt = static_cast<float>(timeStepInSeconds);
    float mass = sph->mass();
    float h = sph->kernelRadius();
    size_t n = sph->numberOfParticles();

    auto d = sph->densities();
    auto p = sph->pressures();
    auto x = sph->positions();
    auto v = sph->velocities();
    auto s = smoothedVelocities();
    auto f = forces();

    auto xs = tempPositions();
    auto vs = tempVelocities();
    auto pf = pressureForces();
    auto ds = tempDensities();
    auto de = densityErrors();

    float targetDensity = sph->targetDensity();
    float delta = computeDelta(dt);

    float factor = dt * pseudoViscosityCoefficient();
    factor = clamp(factor, 0.0f, 1.0f);

    // Build neighbor searcher
    sph->buildNeighborSearcher();
    sph->buildNeighborListsAndUpdateDensities();
    auto ns = sph->neighborStarts();
    auto ne = sph->neighborEnds();
    auto nl = sph->neighborLists();

    // Initialize buffers and compute non-pressure forces
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),

        InitializeBuffersAndComputeForces(
            mass, h, toFloat4(gravity(), 0.0f), viscosityCoefficient(),
            ns.data(), ne.data(), nl.data(), x.data(), v.data(), s.data(),
            f.data(), d.data(), p.data(), pf.data(), ds.data(), de.data()));

    // Prediction-correction
    // unsigned int maxNumIter = 0;
    // float maxDensityError;
    // float densityErrorRatio = 0.0f;

    for (unsigned int k = 0; k < _maxNumberOfIterations; ++k) {
        // Predict velocity / position and resolve collisions
        thrust::for_each(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(n),

            TimeIntegration(dt, 0.0f, x.data(), v.data(), xs.data(), vs.data(),
                            s.data(), f.data(), pf.data()));

        // Compute pressure from density error
        thrust::for_each(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(n),

            ComputeDensityError(mass, h, targetDensity, delta,
                                negativePressureScale(), ns.data(), ne.data(),
                                nl.data(), xs.data(), d.data(), p.data(),
                                de.data(), ds.data()));

        // Compute pressure gradient force
        thrust::for_each(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(n),

            ComputePressureForces(mass, h, ns.data(), ne.data(), nl.data(),
                                  x.data(), pf.data(), ds.data(), p.data()));

        // Compute max density error
    }

    // Accumulate pressure force and time-integrate
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n),

                     TimeIntegration(dt, factor, x.data(), v.data(), x.data(),
                                     v.data(), s.data(), f.data(), pf.data()));
}

#endif  // JET_USE_CUDA
