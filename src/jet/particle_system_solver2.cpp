// Copyright (c) 2016 Doyub Kim

#include <pch.h>

#include <jet/array_utils.h>
#include <jet/constant_vector_field2.h>
#include <jet/parallel.h>
#include <jet/particle_system_solver2.h>

#include <algorithm>

namespace jet {

ParticleSystemSolver2::ParticleSystemSolver2() {
    _particleSystemData = std::make_shared<ParticleSystemData2>();
    _wind = std::make_shared<ConstantVectorField2>(Vector2D());
}

ParticleSystemSolver2::~ParticleSystemSolver2() {
}

double ParticleSystemSolver2::dragCoefficient() const {
    return _dragCoefficient;
}

void ParticleSystemSolver2::setDragCoefficient(double newDragCoefficient) {
    _dragCoefficient = std::max(newDragCoefficient, 0.0);
}

double ParticleSystemSolver2::restitutionCoefficient() const {
    return _restitutionCoefficient;
}

void ParticleSystemSolver2::setRestitutionCoefficient(
    double newRestitutionCoefficient) {
    _restitutionCoefficient = clamp(newRestitutionCoefficient, 0.0, 1.0);
}

const Vector2D& ParticleSystemSolver2::gravity() const {
    return _gravity;
}

void ParticleSystemSolver2::setGravity(const Vector2D& newGravity) {
    _gravity = newGravity;
}

const ParticleSystemData2Ptr&
ParticleSystemSolver2::particleSystemData() const {
    return _particleSystemData;
}

const Collider2Ptr& ParticleSystemSolver2::collider() const {
    return _collider;
}

void ParticleSystemSolver2::setCollider(
    const Collider2Ptr& newCollider) {
    _collider = newCollider;
}

const VectorField2Ptr& ParticleSystemSolver2::wind() const {
    return _wind;
}

void ParticleSystemSolver2::setWind(const VectorField2Ptr& newWind) {
    _wind = newWind;
}

void ParticleSystemSolver2::onAdvanceTimeStep(double timeStepInSeconds) {
    beginAdvanceTimeStep(timeStepInSeconds);

    accumulateForces(timeStepInSeconds);
    timeIntegration(timeStepInSeconds);
    resolveCollision();

    endAdvanceTimeStep(timeStepInSeconds);
}

void ParticleSystemSolver2::accumulateForces(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);

    // Add external forces
    accumulateExternalForces();
}

void ParticleSystemSolver2::beginAdvanceTimeStep(double timeStepInSeconds) {
    // Allocate buffers
    size_t n = _particleSystemData->numberOfParticles();
    _newPositions.resize(n);
    _newVelocities.resize(n);

    // Clear forces
    auto forces = _particleSystemData->forces();
    setRange1(forces.size(), Vector2D(), &forces);

    onBeginAdvanceTimeStep(timeStepInSeconds);
}

void ParticleSystemSolver2::endAdvanceTimeStep(double timeStepInSeconds) {
    // Update data
    size_t n = _particleSystemData->numberOfParticles();
    auto positions = _particleSystemData->positions();
    auto velocities = _particleSystemData->velocities();
    parallelFor(
        kZeroSize,
        n,
        [this, &positions, &velocities](size_t i) {
            positions[i] = _newPositions[i];
            velocities[i] = _newVelocities[i];
        });

    onEndAdvanceTimeStep(timeStepInSeconds);
}

void ParticleSystemSolver2::onBeginAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void ParticleSystemSolver2::onEndAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

void ParticleSystemSolver2::resolveCollision() {
    resolveCollision(
        _newPositions.accessor(),
        _newVelocities.accessor());
}

void ParticleSystemSolver2::resolveCollision(
    ArrayAccessor1<Vector2D> newPositions,
    ArrayAccessor1<Vector2D> newVelocities) {
    if (_collider != nullptr) {
        size_t numberOfParticles = _particleSystemData->numberOfParticles();
        const double radius = _particleSystemData->radius();

        parallelFor(
            kZeroSize,
            numberOfParticles,
            [&](size_t i) {
                _collider->resolveCollision(
                    radius,
                    _restitutionCoefficient,
                    &newPositions[i],
                    &newVelocities[i]);
            });
    }
}

void ParticleSystemSolver2::setParticleSystemData(
    const ParticleSystemData2Ptr& newParticles) {
    _particleSystemData = newParticles;
}

void ParticleSystemSolver2::accumulateExternalForces() {
    size_t n = _particleSystemData->numberOfParticles();
    auto forces = _particleSystemData->forces();
    auto velocities = _particleSystemData->velocities();
    auto positions = _particleSystemData->positions();
    const double mass = _particleSystemData->mass();

    parallelFor(
        kZeroSize,
        n,
        [this, &forces, &velocities, &positions, mass](size_t i) {
            // Gravity
            Vector2D force = mass * _gravity;

            // Wind forces
            Vector2D relativeVel = velocities[i] - _wind->sample(positions[i]);
            force += -_dragCoefficient * relativeVel;

            forces[i] += force;
        });
}

void ParticleSystemSolver2::timeIntegration(double timeStepInSeconds) {
    size_t n = _particleSystemData->numberOfParticles();
    auto forces = _particleSystemData->forces();
    auto velocities = _particleSystemData->velocities();
    auto positions = _particleSystemData->positions();
    const double mass = _particleSystemData->mass();

    parallelFor(
        kZeroSize,
        n,
        [this, timeStepInSeconds, &forces, &velocities, &positions, mass]
        (size_t i) {
            // Integrate velocity first
            Vector2D& newVelocity = _newVelocities[i];
            newVelocity = velocities[i]
                + timeStepInSeconds * forces[i] / mass;

            // Integrate position.
            Vector2D& newPosition = _newPositions[i];
            newPosition = positions[i] + timeStepInSeconds * newVelocity;
        });
}

}  // namespace jet
