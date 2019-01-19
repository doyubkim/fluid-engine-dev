// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/particle_system_solver.h>
#include <jet/timer.h>

namespace jet {

template <size_t N>
ParticleSystemSolver<N>::ParticleSystemSolver() {
    _particleSystemData = std::make_shared<ParticleSystemData<N>>();
}

template <size_t N>
ParticleSystemSolver<N>::~ParticleSystemSolver() {}

template <size_t N>
double ParticleSystemSolver<N>::dragCoefficient() const {
    return _dragCoefficient;
}

template <size_t N>
void ParticleSystemSolver<N>::setDragCoefficient(double newDragCoefficient) {
    _dragCoefficient = std::max(newDragCoefficient, 0.0);
}

template <size_t N>
double ParticleSystemSolver<N>::restitutionCoefficient() const {
    return _restitutionCoefficient;
}

template <size_t N>
void ParticleSystemSolver<N>::setRestitutionCoefficient(
    double newRestitutionCoefficient) {
    _restitutionCoefficient = clamp(newRestitutionCoefficient, 0.0, 1.0);
}

template <size_t N>
const Vector<double, N>& ParticleSystemSolver<N>::gravity() const {
    return _gravity;
}

template <size_t N>
void ParticleSystemSolver<N>::setGravity(const Vector<double, N>& newGravity) {
    _gravity = newGravity;
}

template <size_t N>
const ParticleSystemDataPtr<N>& ParticleSystemSolver<N>::particleSystemData()
    const {
    return _particleSystemData;
}

template <size_t N>
const ColliderPtr<N>& ParticleSystemSolver<N>::collider() const {
    return _collider;
}

template <size_t N>
void ParticleSystemSolver<N>::setCollider(const ColliderPtr<N>& newCollider) {
    _collider = newCollider;
}

template <size_t N>
const ParticleEmitterPtr<N>& ParticleSystemSolver<N>::emitter() const {
    return _emitter;
}

template <size_t N>
void ParticleSystemSolver<N>::setEmitter(
    const ParticleEmitterPtr<N>& newEmitter) {
    _emitter = newEmitter;
    newEmitter->setTarget(_particleSystemData);
}

template <size_t N>
const VectorFieldPtr<N>& ParticleSystemSolver<N>::wind() const {
    return _wind;
}

template <size_t N>
void ParticleSystemSolver<N>::setWind(const VectorFieldPtr<N>& newWind) {
    _wind = newWind;
}

template <size_t N>
void ParticleSystemSolver<N>::onInitialize() {
    // When initializing the solver, update the collider and emitter state
    // as well since they also affects the initial condition of the
    // simulation.
    Timer timer;
    updateCollider(0.0);
    JET_INFO << "Update collider took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    updateEmitter(0.0);
    JET_INFO << "Update emitter took " << timer.durationInSeconds()
             << " seconds";
}

template <size_t N>
void ParticleSystemSolver<N>::onAdvanceTimeStep(double timeStepInSeconds) {
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

template <size_t N>
void ParticleSystemSolver<N>::accumulateForces(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

template <size_t N>
void ParticleSystemSolver<N>::onBeginAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

template <size_t N>
void ParticleSystemSolver<N>::onEndAdvanceTimeStep(double timeStepInSeconds) {
    UNUSED_VARIABLE(timeStepInSeconds);
}

template <size_t N>
void ParticleSystemSolver<N>::resolveCollision() {
    auto x = _particleSystemData->positions();
    auto v = _particleSystemData->velocities();
    resolveCollision(x, v);
}

template <size_t N>
void ParticleSystemSolver<N>::resolveCollision(
    ArrayView1<Vector<double, N>> newPositions,
    ArrayView1<Vector<double, N>> newVelocities) {
    if (_collider != nullptr) {
        // TODO: Implement vectorized version
//        const double radius = _particleSystemData->radius();
//        _collider->resolveCollision(radius, _restitutionCoefficient,
//                                    newPositions, newVelocities);
    }
}

template <size_t N>
void ParticleSystemSolver<N>::setParticleSystemData(
    const ParticleSystemDataPtr<N>& newParticles) {
    _particleSystemData = newParticles;
}

template <size_t N>
void ParticleSystemSolver<N>::updateCollider(double timeStepInSeconds) {
    if (_collider != nullptr) {
        _collider->update(currentTimeInSeconds(), timeStepInSeconds);
    }
}

template <size_t N>
void ParticleSystemSolver<N>::updateEmitter(double timeStepInSeconds) {
    if (_emitter != nullptr) {
        _emitter->update(currentTimeInSeconds(), timeStepInSeconds);
    }
}

template <size_t N>
void ParticleSystemSolver<N>::beginAdvanceTimeStep(double timeStepInSeconds) {
    // Clear forces
    auto forces = particleSystemData()->forces();
    forces.fill(Vector<double, N>());

    // Update collider and emitter
    Timer timer;
    updateCollider(timeStepInSeconds);
    JET_INFO << "Update collider took " << timer.durationInSeconds()
             << " seconds";

    timer.reset();
    updateEmitter(timeStepInSeconds);
    JET_INFO << "Update emitter took " << timer.durationInSeconds()
             << " seconds";

    onBeginAdvanceTimeStep(timeStepInSeconds);
}

template <size_t N>
void ParticleSystemSolver<N>::endAdvanceTimeStep(double timeStepInSeconds) {
    onEndAdvanceTimeStep(timeStepInSeconds);
}

template <size_t N>
void ParticleSystemSolver<N>::timeIntegration(double dt) {
    const auto& particles = particleSystemData();
    size_t n = particles->numberOfParticles();
    auto f = particles->forces();
    auto v = particles->velocities();
    auto x = particles->positions();

    const double m = particles->mass();
    const Vector<double, N> g = gravity();
    const double drag = dragCoefficient();

    // Sample wind
    Array1<Vector<double, N>> windValues;
    // TODO: Implement vectorized version
//    wind()->getSamples(x, windValues);

    // Integration
    parallelFor(kZeroSize, n, [&](size_t i) {
        Vector<double, N> force;

        // Gravity
        force += m * g;

        // Wind force
        Vector<double, N> relativeVel = v[i] - windValues[i];
        force -= drag * relativeVel;

        // Other forces
        force += f[i];

        // Integrate velocity first
        Vector<double, N> newVelocity = v[i] + dt * force / m;

        // Integrate position.
        Vector<double, N> newPosition = x[i] + dt * newVelocity;

        v[i] = newVelocity;
        x[i] = newPosition;
    });
}

template class ParticleSystemSolver<2>;

template class ParticleSystemSolver<3>;

}  // namespace jet
