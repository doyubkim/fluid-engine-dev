// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/cuda_particle_system_solver2.h>
#include <jet/cuda_utils.h>
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

struct ResolveCollision {
    ResolveCollision() {}

    JET_CUDA_DEVICE void operator()(float2* newPosition, float2* newVelocity) {
        // TODO: Use collider
        if (newPosition->y < 0.0f) {
            newPosition->y = 0.0f;
            if (newVelocity->y < 0.0f) {
                newVelocity->y *= -1.0;
            }
        }
    }
};

struct AccumulateExternalForces {
    float mass;
    float2 gravity;

    AccumulateExternalForces(float m, float2 g) : mass(m), gravity(g) {}

    JET_CUDA_DEVICE float2 operator()(const float2& initialForce) {
        // Gravity
        float2 force = mass * gravity + initialForce;
        return force;
    }
};

struct TimeIntegration {
    float mass;
    float timeStepInSeconds;

    TimeIntegration(float m, float dt) : mass(m), timeStepInSeconds(dt) {}

    JET_CUDA_DEVICE void operator()(const float2& p0, const float2& v0,
                                    const float2& f, float2* p1,
                                    float2* v1) const {
        *v1 = v0 + timeStepInSeconds * f / mass;
        *p1 = p0 + timeStepInSeconds * (*v1);
    }
};

struct AdvanceTimeStepKernel {
    AccumulateExternalForces accExtForces;
    TimeIntegration ti;
    ResolveCollision rc;

    AdvanceTimeStepKernel(float m, float dt, float2 gravity)
        : accExtForces(m, gravity), ti(m, dt) {}

    template <typename Tuple>
    JET_CUDA_DEVICE void operator()(const Tuple& t) {
        // posCurr: 0
        // velCurr: 1
        float2 p0 = get<0>(t);
        float2 v0 = get<1>(t);
        float2 p1;
        float2 v1;

        float2 f = accExtForces(make_float2(0, 0));
        ti(p0, v0, f, &p1, &v1);
        rc(&p1, &v1);

        get<0>(t) = p1;
        get<1>(t) = v1;
    }
};

}  // namespace

void CudaParticleSystemSolver2::onAdvanceTimeStep(double timeStepInSeconds) {
    updateCollider(timeStepInSeconds);
    updateEmitter(timeStepInSeconds);

    auto particles = particleSystemData();
    auto posCurr = particles->positions();
    auto velCurr = particles->velocities();
    auto dt = static_cast<float>(timeStepInSeconds);
    auto g = toFloat2(gravity());

    AdvanceTimeStepKernel kernel(_mass, dt, g);

    thrust::for_each(
        make_zip_iterator(make_tuple(posCurr.begin(), velCurr.begin())),
        make_zip_iterator(make_tuple(posCurr.end(), velCurr.end())), kernel);
}
