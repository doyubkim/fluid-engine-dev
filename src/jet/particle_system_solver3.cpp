// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_utils.h>
#include <jet/constant_vector_field3.h>
#include <jet/parallel.h>
#include <jet/particle_system_solver3.h>
#include <jet/timer.h>

#include <algorithm>

namespace jet {

ParticleSystemSolver3::ParticleSystemSolver3()
: ParticleSystemSolver3(1e-3, 1e-3) {
}

ParticleSystemSolver3::ParticleSystemSolver3(
    double radius,
    double mass) {
    _particleSystemData = std::make_shared<ParticleSystemData3>();
    _particleSystemData->setRadius(radius);
    _particleSystemData->setMass(mass);
    _wind = std::make_shared<ConstantVectorField3>(Vector3D());
}

ParticleSystemSolver3::~ParticleSystemSolver3() {
}

double ParticleSystemSolver3::dragCoefficient() const {
    return _dragCoefficient;
}

void ParticleSystemSolver3::setDragCoefficient(double newDragCoefficient) {
    _dragCoefficient = std::max(newDragCoefficient, 0.0);
}

double ParticleSystemSolver3::restitutionCoefficient() const {
    return _restitutionCoefficient;
}

void ParticleSystemSolver3::setRestitutionCoefficient(
    double newRestitutionCoefficient) {
    _restitutionCoefficient = clamp(newRestitutionCoefficient, 0.0, 1.0);
}

const Vector3D& ParticleSystemSolver3::gravity() const {
    return _gravity;
}

void ParticleSystemSolver3::setGravity(const Vector3D& newGravity) {
    _gravity = newGravity;
}

const ParticleSystemData3Ptr&
ParticleSystemSolver3::particleSystemData() const {
    return _particleSystemData;
}

const Collider3Ptr& ParticleSystemSolver3::collider() const {
    return _collider;
}

void ParticleSystemSolver3::setCollider(
    const Collider3Ptr& newCollider) {
    _collider = newCollider;
}

const ParticleEmitter3Ptr& ParticleSystemSolver3::emitter() const {
    return _emitter;
}

void ParticleSystemSolver3::setEmitter(
    const ParticleEmitter3Ptr& newEmitter) {
    _emitter = newEmitter;
    newEmitter->setTarget(_particleSystemData);
}

const VectorField3Ptr& ParticleSystemSolver3::wind() const {
    return _wind;
}

void ParticleSystemSolver3::setWind(const VectorField3Ptr& newWind) {
    _wind = newWind;
}

void ParticleSystemSolver3::onInitialize() {
    // When initializing the solver, update the collider and emitter state as
    // well since they also affects the initial condition of the simulation.
    Timer timer;
    updateCollider(0.0);
    JET_INFO << "Update collider took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    updateEmitter(0.0);
    JET_INFO << "Update emitter took "
             << timer.durationInSeconds() << " seconds";
}

void ParticleSystemSolver3::onAdvanceTimeStep(double timeStepInSeconds) {
    beginAdvanceTimeStep(timeStepInSeconds);

    Timer timer;
    accumulateForces(timeStepInSeconds);
    JET_INFO << "Accumulating forces took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    timeIntegration(timeStepInSeconds);
    JET_INFO << "Time integration took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    resolveCollision();
    JET_INFO << "Resolving collision took "
             << timer.durationInSeconds() << " seconds";

    endAdvanceTimeStep(timeStepInSeconds);
}

void ParticleSystemSolver3::accumulateForces(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);

    // Add external forces
    accumulateExternalForces();
}

void ParticleSystemSolver3::beginAdvanceTimeStep(double timeStepInSeconds) {
    // Clear forces
    auto forces = _particleSystemData->forces();
    setRange1(forces.size(), Vector3D(), &forces);

    // Update collider and emitter
    Timer timer;
    updateCollider(timeStepInSeconds);
    JET_INFO << "Update collider took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    updateEmitter(timeStepInSeconds);
    JET_INFO << "Update emitter took "
             << timer.durationInSeconds() << " seconds";

    // Allocate buffers
    size_t n = _particleSystemData->numberOfParticles();
    _newPositions.resize(n);
    _newVelocities.resize(n);

    onBeginAdvanceTimeStep(timeStepInSeconds);
}

void ParticleSystemSolver3::endAdvanceTimeStep(double timeStepInSeconds) {
    // Update data
    size_t n = _particleSystemData->numberOfParticles();
    auto positions = _particleSystemData->positions();
    auto velocities = _particleSystemData->velocities();
    parallelFor(
        kZeroSize,
        n,
        [&] (size_t i) {
            positions[i] = _newPositions[i];
            velocities[i] = _newVelocities[i];
        });

    onEndAdvanceTimeStep(timeStepInSeconds);
}

void ParticleSystemSolver3::onBeginAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void ParticleSystemSolver3::onEndAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void ParticleSystemSolver3::resolveCollision() {
    resolveCollision(
        _newPositions.accessor(),
        _newVelocities.accessor());
}

void ParticleSystemSolver3::resolveCollision(
    ArrayAccessor1<Vector3D> newPositions,
    ArrayAccessor1<Vector3D> newVelocities) {
    if (_collider != nullptr) {
        size_t numberOfParticles = _particleSystemData->numberOfParticles();
        const double radius = _particleSystemData->radius();

        parallelFor(
            kZeroSize,
            numberOfParticles,
            [&] (size_t i) {
                _collider->resolveCollision(
                    radius,
                    _restitutionCoefficient,
                    &newPositions[i],
                    &newVelocities[i]);
            });
    }
}

void ParticleSystemSolver3::setParticleSystemData(
    const ParticleSystemData3Ptr& newParticles) {
    _particleSystemData = newParticles;
}

void ParticleSystemSolver3::accumulateExternalForces() {
    size_t n = _particleSystemData->numberOfParticles();
    auto forces = _particleSystemData->forces();
    auto velocities = _particleSystemData->velocities();
    auto positions = _particleSystemData->positions();
    const double mass = _particleSystemData->mass();

    parallelFor(
        kZeroSize,
        n,
        [&] (size_t i) {
            // Gravity
            Vector3D force = mass * _gravity;

            // Wind forces
            Vector3D relativeVel = velocities[i] - _wind->sample(positions[i]);
            force += -_dragCoefficient * relativeVel;

            forces[i] += force;
        });
}

void ParticleSystemSolver3::timeIntegration(double timeStepInSeconds) {
    size_t n = _particleSystemData->numberOfParticles();
    auto forces = _particleSystemData->forces();
    auto velocities = _particleSystemData->velocities();
    auto positions = _particleSystemData->positions();
    const double mass = _particleSystemData->mass();

    parallelFor(
        kZeroSize,
        n,
        [&] (size_t i) {
            // Integrate velocity first
            Vector3D& newVelocity = _newVelocities[i];
            newVelocity = velocities[i]
                + timeStepInSeconds * forces[i] / mass;

            // Integrate position.
            Vector3D& newPosition = _newPositions[i];
            newPosition = positions[i] + timeStepInSeconds * newVelocity;
        });
}

void ParticleSystemSolver3::updateCollider(double timeStepInSeconds) {
    if (_collider != nullptr) {
        _collider->update(currentTimeInSeconds(), timeStepInSeconds);
    }
}

void ParticleSystemSolver3::updateEmitter(double timeStepInSeconds) {
    if (_emitter != nullptr) {
        _emitter->update(currentTimeInSeconds(), timeStepInSeconds);
    }
}

ParticleSystemSolver3::Builder ParticleSystemSolver3::builder() {
    return Builder();
}


ParticleSystemSolver3 ParticleSystemSolver3::Builder::build() const {
    return ParticleSystemSolver3(_radius, _mass);
}

ParticleSystemSolver3Ptr ParticleSystemSolver3::Builder::makeShared() const {
    return std::shared_ptr<ParticleSystemSolver3>(
        new ParticleSystemSolver3(_radius, _mass),
        [] (ParticleSystemSolver3* obj) {
            delete obj;
        });
}

}  // namespace jet
