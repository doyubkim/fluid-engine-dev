// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include <pch.h>

#include <jet/cuda_particle_system_solver_base3.h>

using namespace jet;

CudaParticleSystemSolverBase3::CudaParticleSystemSolverBase3() {
    _particleSystemData = std::make_shared<CudaParticleSystemData3>();
}

CudaParticleSystemSolverBase3::~CudaParticleSystemSolverBase3() {}

float CudaParticleSystemSolverBase3::dragCoefficient() const {
    return _dragCoefficient;
}

void CudaParticleSystemSolverBase3::setDragCoefficient(float newDragCoefficient) {
    _dragCoefficient = std::max(newDragCoefficient, 0.0f);
}

float CudaParticleSystemSolverBase3::restitutionCoefficient() const {
    return _restitutionCoefficient;
}

void CudaParticleSystemSolverBase3::setRestitutionCoefficient(
    float newRestitutionCoefficient) {
    _restitutionCoefficient = clamp(newRestitutionCoefficient, 0.0f, 1.0f);
}

const Vector3F& CudaParticleSystemSolverBase3::gravity() const { return _gravity; }

void CudaParticleSystemSolverBase3::setGravity(const Vector3F& newGravity) {
    _gravity = newGravity;
}

CudaParticleSystemData3*
CudaParticleSystemSolverBase3::particleSystemData() {
    return _particleSystemData.get();
}

const CudaParticleSystemData3*
CudaParticleSystemSolverBase3::particleSystemData() const {
    return _particleSystemData.get();
}

void CudaParticleSystemSolverBase3::onInitialize() {
    // When initializing the solver, update the collider and emitter state as
    // well since they also affects the initial condition of the simulation.
    updateCollider(0.0f);
    updateEmitter(0.0f);
}

void CudaParticleSystemSolverBase3::updateCollider(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void CudaParticleSystemSolverBase3::updateEmitter(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

#endif  // JET_USE_CUDA
