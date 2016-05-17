// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PARTICLE_SYSTEM_SOLVER2_H_
#define INCLUDE_JET_PARTICLE_SYSTEM_SOLVER2_H_

#include <jet/collider2.h>
#include <jet/constants.h>
#include <jet/vector_field2.h>
#include <jet/particle_system_data2.h>
#include <jet/physics_animation.h>

namespace jet {

class ParticleSystemSolver2 : public PhysicsAnimation {
 public:
    ParticleSystemSolver2();

    virtual ~ParticleSystemSolver2();

    double dragCoefficient() const;

    void setDragCoefficient(double newDragCoefficient);

    double restitutionCoefficient() const;

    void setRestitutionCoefficient(double newRestitutionCoefficient);

    const Vector2D& gravity() const;

    void setGravity(const Vector2D& newGravity);

    const ParticleSystemData2Ptr& particleSystemData() const;

    const Collider2Ptr& collider() const;

    void setCollider(const Collider2Ptr& newCollider);

    const VectorField2Ptr& wind() const;

    void setWind(const VectorField2Ptr& newWind);

 protected:
    void onAdvanceTimeStep(double timeStepInSeconds) override;

    virtual void accumulateForces(double timeStepInSeconds);

    virtual void onBeginAdvanceTimeStep(double timeStepInSeconds);

    virtual void onEndAdvanceTimeStep(double timeStepInSeconds);

    void resolveCollision();

    void resolveCollision(
        ArrayAccessor1<Vector2D> newPositions,
        ArrayAccessor1<Vector2D> newVelocities);

    void setParticleSystemData(const ParticleSystemData2Ptr& newParticles);

 private:
    double _dragCoefficient = 1e-4;
    double _restitutionCoefficient = 0.0;
    Vector2D _gravity = Vector2D(0.0, kGravity);

    ParticleSystemData2Ptr _particleSystemData;
    ParticleSystemData2::VectorData _newPositions;
    ParticleSystemData2::VectorData _newVelocities;
    Collider2Ptr _collider;
    VectorField2Ptr _wind;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);

    void accumulateExternalForces();

    void timeIntegration(double timeStepInSeconds);
};

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_SYSTEM_SOLVER2_H_
