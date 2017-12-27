// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/cuda_particle_system_solver3.h>
#include <jet/timer.h>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>

#include <helper_math.h>

#include <algorithm>

using namespace jet;
using namespace experimental;
using thrust::get;
using thrust::make_tuple;
using thrust::make_zip_iterator;

namespace {

struct UpdateStateVectors {
    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        get<0>(t) = get<1>(t);
        get<2>(t) = get<3>(t);
    }
};

struct ResolveCollision {
    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        // TODO: Replace with proper collider
        float4 x = get<0>(t);
        float4 v = get<1>(t);

        if (x.y < 0.0) {
            x.y = 0.0;
            v.y *= -1.0;
        }

        get<0>(t) = x;
        get<1>(t) = v;
    }
};

struct AccumulateExternalForces {
    float mass;
    float4 gravity;

    AccumulateExternalForces(float m, float4 g) : mass(m), gravity(g) {}

    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        // Gravity
        float4 force = mass * gravity;

        // // Wind forces
        // Vector3F relativeVel = velocities[i] -
        // _wind->sample(positions[i]); force += -_dragCoefficient *
        // relativeVel;

        get<2>(t) += force;
    }
};

struct TimeIntegration {
    float mass;
    float timeStepInSeconds;

    TimeIntegration(float m, float dt) : mass(m), timeStepInSeconds(dt) {}

    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        // Integrate velocity first
        get<3>(t) = get<2>(t) + timeStepInSeconds * get<4>(t) / mass;

        // Integrate position.
        get<1>(t) = get<0>(t) + timeStepInSeconds * get<3>(t);
    }
};

}  // namespace

CudaParticleSystemSolver3::CudaParticleSystemSolver3()
    : CudaParticleSystemSolver3(1e-3f, 1e-3f) {}

CudaParticleSystemSolver3::CudaParticleSystemSolver3(float radius, float mass)
    : _radius(radius), _mass(mass) {
    _particleSystemData = std::make_shared<CudaParticleSystemData3>();
    _forcesIdx = _particleSystemData->addVectorData();
    _newPositionsIdx = _particleSystemData->addVectorData();
    _newVelocitiesIdx = _particleSystemData->addVectorData();
}

CudaParticleSystemSolver3::~CudaParticleSystemSolver3() {}

float CudaParticleSystemSolver3::dragCoefficient() const {
    return _dragCoefficient;
}

void CudaParticleSystemSolver3::setDragCoefficient(float newDragCoefficient) {
    _dragCoefficient = std::max(newDragCoefficient, 0.0f);
}

float CudaParticleSystemSolver3::restitutionCoefficient() const {
    return _restitutionCoefficient;
}

void CudaParticleSystemSolver3::setRestitutionCoefficient(
    float newRestitutionCoefficient) {
    _restitutionCoefficient = clamp(newRestitutionCoefficient, 0.0f, 1.0f);
}

const Vector3F& CudaParticleSystemSolver3::gravity() const { return _gravity; }

void CudaParticleSystemSolver3::setGravity(const Vector3F& newGravity) {
    _gravity = newGravity;
}

const CudaParticleSystemData3Ptr&
CudaParticleSystemSolver3::particleSystemData() const {
    return _particleSystemData;
}

void CudaParticleSystemSolver3::onInitialize() {
    // When initializing the solver, update the collider and emitter state as
    // well since they also affects the initial condition of the simulation.
    Timer timer;
    updateCollider(0.0f);
    JET_INFO << "Update collider took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    updateEmitter(0.0f);
    JET_INFO << "Update emitter took " << timer.durationInSeconds()
             << " seconds";
}

void CudaParticleSystemSolver3::onAdvanceTimeStep(double timeStepInSeconds) {
    beginAdvanceTimeStep(timeStepInSeconds);

    Timer timer;
    accumulateForces(timeStepInSeconds);
    JET_INFO << "Accumulating forces took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    timeIntegration(timeStepInSeconds);
    JET_INFO << "Time integration took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    resolveCollision();
    JET_INFO << "Resolving collision took " << timer.durationInSeconds()
             << " seconds";

    endAdvanceTimeStep(timeStepInSeconds);
}

void CudaParticleSystemSolver3::accumulateForces(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);

    // Add external forces
    accumulateExternalForces();
}

void CudaParticleSystemSolver3::beginAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
    // Clear forces
    auto forces = _particleSystemData->vectorDataAt(_forcesIdx);
    thrust::fill(forces.begin(), forces.end(), make_float4(0, 0, 0, 0));

    onBeginAdvanceTimeStep(timeStepInSeconds);
}

void CudaParticleSystemSolver3::endAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);

    auto posCurr = _particleSystemData->positions();
    auto velCurr = _particleSystemData->velocities();
    auto posNew = _particleSystemData->vectorDataAt(_newPositionsIdx);
    auto velNew = _particleSystemData->vectorDataAt(_newVelocitiesIdx);

    thrust::for_each(
        thrust::device,
        make_zip_iterator(make_tuple(posCurr.begin(), posNew.begin(),
                                     velCurr.begin(), velNew.begin())),
        make_zip_iterator(make_tuple(posCurr.end(), posNew.end(), velCurr.end(),
                                     velNew.end())),
        UpdateStateVectors());

    onEndAdvanceTimeStep(timeStepInSeconds);
}

void CudaParticleSystemSolver3::onBeginAdvanceTimeStep(
    double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void CudaParticleSystemSolver3::onEndAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void CudaParticleSystemSolver3::resolveCollision() {
    resolveCollision(_particleSystemData->vectorDataAt(_newPositionsIdx),
                     _particleSystemData->vectorDataAt(_newVelocitiesIdx));
}

void CudaParticleSystemSolver3::resolveCollision(
    CudaArrayView1<float4> newPositions, CudaArrayView1<float4> newVelocities) {
    thrust::for_each(
        thrust::device,
        make_zip_iterator(
            make_tuple(newPositions.begin(), newVelocities.begin())),
        make_zip_iterator(make_tuple(newPositions.end(), newVelocities.end())),
        ResolveCollision());
}

void CudaParticleSystemSolver3::accumulateExternalForces() {
    auto pos = _particleSystemData->positions();
    auto vel = _particleSystemData->velocities();
    auto forces = _particleSystemData->vectorDataAt(_forcesIdx);

    thrust::for_each(
        thrust::device,
        make_zip_iterator(make_tuple(pos.begin(), vel.begin(), forces.begin())),
        make_zip_iterator(make_tuple(pos.end(), vel.end(), forces.end())),
        AccumulateExternalForces(
            _mass, make_float4(_gravity.x, _gravity.y, _gravity.z, 0.0f)));
}

void CudaParticleSystemSolver3::timeIntegration(double timeStepInSeconds) {
    auto posCurr = _particleSystemData->positions();
    auto velCurr = _particleSystemData->velocities();
    auto posNew = _particleSystemData->vectorDataAt(_newPositionsIdx);
    auto velNew = _particleSystemData->vectorDataAt(_newVelocitiesIdx);
    auto forces = _particleSystemData->vectorDataAt(_forcesIdx);

    thrust::for_each(
        thrust::device,
        make_zip_iterator(make_tuple(posCurr.begin(), posNew.begin(),
                                     velCurr.begin(), velNew.begin(),
                                     forces.begin())),
        make_zip_iterator(make_tuple(posCurr.end(), posNew.end(), velCurr.end(),
                                     velNew.end(), forces.end())),
        TimeIntegration(_mass, static_cast<float>(timeStepInSeconds)));
}

void CudaParticleSystemSolver3::updateCollider(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
    // if (_collider != nullptr) {
    //     _collider->update(currentTimeInSeconds(), timeStepInSeconds);
    // }
}

void CudaParticleSystemSolver3::updateEmitter(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
    // if (_emitter != nullptr) {
    //     _emitter->update(currentTimeInSeconds(), timeStepInSeconds);
    // }
}

CudaParticleSystemSolver3::Builder CudaParticleSystemSolver3::builder() {
    return Builder();
}

CudaParticleSystemSolver3 CudaParticleSystemSolver3::Builder::build() const {
    return CudaParticleSystemSolver3(_radius, _mass);
}

CudaParticleSystemSolver3Ptr CudaParticleSystemSolver3::Builder::makeShared()
    const {
    return std::shared_ptr<CudaParticleSystemSolver3>(
        new CudaParticleSystemSolver3(_radius, _mass),
        [](CudaParticleSystemSolver3* obj) { delete obj; });
}
