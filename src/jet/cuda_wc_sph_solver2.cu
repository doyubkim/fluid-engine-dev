// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constants.h>
#include <jet/cuda_sph_kernels2.h>
#include <jet/cuda_utils.h>
#include <jet/cuda_wc_sph_solver2.h>
#include <jet/timer.h>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>

#include <algorithm>

using namespace jet;
using namespace experimental;
using thrust::get;
using thrust::make_tuple;
using thrust::make_zip_iterator;

namespace {

class ComputePressureFunc {
 public:
    inline ComputePressureFunc(float targetDensity, float eosScale,
                               float eosExponent, float negativePressureScale)
        : _targetDensity(targetDensity),
          _eosScale(eosScale),
          _eosExponent(eosExponent),
          _negativePressureScale(negativePressureScale) {}

    template <typename Float>
    inline JET_CUDA_HOST_DEVICE float operator()(Float d) {
        return computePressureFromEos(d, _targetDensity, _eosScale,
                                      _eosExponent, _negativePressureScale);
    }

    template <typename Float>
    inline JET_CUDA_HOST_DEVICE float computePressureFromEos(
        Float density, float targetDensity, float eosScale, float eosExponent,
        float negativePressureScale) {
        // Equation of state
        // (http://www.ifi.uzh.ch/vmml/publications/pcisph/pcisph.pdf)
        float p = eosScale / eosExponent *
                  (powf((density / targetDensity), eosExponent) - 1.0f);

        // Negative pressure scaling
        if (p < 0) {
            p *= negativePressureScale;
        }

        return p;
    }

 private:
    float _targetDensity;
    float _eosScale;
    float _eosExponent;
    float _negativePressureScale;
};

class ComputeForces {
 public:
    inline ComputeForces(float m, float h, float2 gravity, float viscosity,
                         uint32_t* neighborStarts, uint32_t* neighborEnds,
                         uint32_t* neighborLists, float2* positions,
                         float2* velocities, float2* smoothedVelocities,
                         float2* forces, float* densities, float* pressures)
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
          _pressures(pressures) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index i) {
        uint32_t ns = _neighborStarts[i];
        uint32_t ne = _neighborEnds[i];

        float2 x_i = _positions[i];
        float2 v_i = _velocities[i];
        float d_i = _densities[i];
        float p_i = _pressures[i];
        float2 f = _gravity;

        float w_i = _mass / d_i * _spikyKernel(0.0f);
        float weightSum = w_i;
        float2 smoothedVelocity = w_i * v_i;

        for (uint32_t jj = ns; jj < ne; ++jj) {
            uint32_t j = _neighborLists[jj];

            float2 r = _positions[j] - x_i;
            float dist = length(r);

            if (dist > 0.0f) {
                float2 dir = r / dist;

                float2 v_j = _velocities[j];
                float d_j = _densities[j];
                float p_j = _pressures[j];

                // Pressure force
                f -= _massSquared * (p_i / (d_i * d_i) + p_j / (d_j * d_j)) *
                     _spikyKernel.gradient(dist, dir);

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
    float2 _gravity;
    float _viscosity;
    CudaSphSpikyKernel2 _spikyKernel;
    uint32_t* _neighborStarts;
    uint32_t* _neighborEnds;
    uint32_t* _neighborLists;
    float2* _positions;
    float2* _velocities;
    float2* _smoothedVelocities;
    float2* _forces;
    float* _densities;
    float* _pressures;
};

#define LOWER_X 0.0f
#define UPPER_X 1.0f
#define LOWER_Y 0.0f
#define UPPER_Y 1.0f
#define BND_R 0.0f

class TimeIntegration {
 public:
    TimeIntegration(float dt, float m, float smoothFactor, float2* positions,
                    float2* velocities, float2* smoothedVelocities,
                    float2* forces)
        : _dt(dt),
          _mass(m),
          _smoothFactor(smoothFactor),
          _positions(positions),
          _velocities(velocities),
          _smoothedVelocities(smoothedVelocities),
          _forces(forces) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index i) {
        float2 x = _positions[i];
        float2 v = _velocities[i];
        float2 s = _smoothedVelocities[i];
        float2 f = _forces[i];

        v = (1.0f - _smoothFactor) * v + _smoothFactor * s;
        v += _dt * f / _mass;
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

        _positions[i] = x;
        _velocities[i] = v;
    }

 private:
    float _dt;
    float _mass;
    float _smoothFactor;
    float2* _positions;
    float2* _velocities;
    float2* _smoothedVelocities;
    float2* _forces;
};

}  // namespace

void CudaWcSphSolver2::onAdvanceTimeStep(double timeStepInSeconds) {
    auto sph = sphSystemData();

    // Build neighbor searcher
    sph->buildNeighborSearcher();
    sph->buildNeighborListsAndUpdateDensities();

    // Compute pressure
    auto d = sph->densities();
    auto p = sph->pressures();
    const float targetDensity = sph->targetDensity();
    const float eosScale =
        targetDensity * square(speedOfSound()) / _eosExponent;
    thrust::transform(
        d.begin(), d.end(), p.begin(),
        ComputePressureFunc(targetDensity, eosScale, eosExponent(),
                            negativePressureScale()));

    // Compute pressure / viscosity forces and smoothed velocity
    size_t n = sph->numberOfParticles();
    float mass = sph->mass();
    float h = sph->kernelRadius();
    auto ns = sph->neighborStarts();
    auto ne = sph->neighborEnds();
    auto nl = sph->neighborLists();
    auto x = sph->positions();
    auto v = sph->velocities();
    auto s = smoothedVelocities();
    auto f = forces();

    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),

        ComputeForces(mass, h, toFloat2(gravity()), viscosityCoefficient(),
                      ns.data(), ne.data(), nl.data(), x.data(), v.data(),
                      s.data(), f.data(), d.data(), p.data()));

    // Time-integration
    float dt = static_cast<float>(timeStepInSeconds);
    float factor = dt * pseudoViscosityCoefficient();
    factor = clamp(factor, 0.0f, 1.0f);

    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n),

                     TimeIntegration(dt, mass, factor, x.data(), v.data(),
                                     s.data(), f.data()));
}
