// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include <pch.h>

#include <jet/cuda_particle_system_solver_base2.h>

using namespace jet;
using namespace experimental;

CudaParticleSystemSolverBase2::CudaParticleSystemSolverBase2() {
    _particleSystemData = std::make_shared<CudaParticleSystemData2>();
}

CudaParticleSystemSolverBase2::~CudaParticleSystemSolverBase2() {}

float CudaParticleSystemSolverBase2::dragCoefficient() const {
    return _dragCoefficient;
}

void CudaParticleSystemSolverBase2::setDragCoefficient(
    float newDragCoefficient) {
    _dragCoefficient = std::max(newDragCoefficient, 0.0f);
}

float CudaParticleSystemSolverBase2::restitutionCoefficient() const {
    return _restitutionCoefficient;
}

void CudaParticleSystemSolverBase2::setRestitutionCoefficient(
    float newRestitutionCoefficient) {
    _restitutionCoefficient = clamp(newRestitutionCoefficient, 0.0f, 1.0f);
}

const Vector2F& CudaParticleSystemSolverBase2::gravity() const {
    return _gravity;
}

void CudaParticleSystemSolverBase2::setGravity(const Vector2F& newGravity) {
    _gravity = newGravity;
}

CudaParticleSystemData2* CudaParticleSystemSolverBase2::particleSystemData() {
    return _particleSystemData.get();
}

const CudaParticleSystemData2*
CudaParticleSystemSolverBase2::particleSystemData() const {
    return _particleSystemData.get();
}

void CudaParticleSystemSolverBase2::onInitialize() {
    // When initializing the solver, update the collider and emitter state as
    // well since they also affects the initial condition of the simulation.
    updateCollider(0.0f);
    updateEmitter(0.0f);
}

void CudaParticleSystemSolverBase2::updateCollider(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void CudaParticleSystemSolverBase2::updateEmitter(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

#endif  // JET_USE_CUDA
