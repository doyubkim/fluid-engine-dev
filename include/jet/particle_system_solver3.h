// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PARTICLE_SYSTEM_SOLVER3_H_
#define INCLUDE_JET_PARTICLE_SYSTEM_SOLVER3_H_

#include <jet/collider3.h>
#include <jet/constants.h>
#include <jet/vector_field3.h>
#include <jet/particle_system_data3.h>
#include <jet/physics_animation.h>

namespace jet {

class ParticleSystemSolver3 : public PhysicsAnimation {
 public:
    ParticleSystemSolver3();

    virtual ~ParticleSystemSolver3();

    double dragCoefficient() const;

    void setDragCoefficient(double newDragCoefficient);

    double restitutionCoefficient() const;

    void setRestitutionCoefficient(double newRestitutionCoefficient);

    const Vector3D& gravity() const;

    void setGravity(const Vector3D& newGravity);

    const ParticleSystemData3Ptr& particleSystemData() const;

    const Collider3Ptr& collider() const;

    void setCollider(const Collider3Ptr& newCollider);

    const VectorField3Ptr& wind() const;

    void setWind(const VectorField3Ptr& newWind);

 protected:
    void onAdvanceTimeStep(double timeStepInSeconds) override;

    virtual void accumulateForces(double timeStepInSeconds);

    virtual void onBeginAdvanceTimeStep(double timeStepInSeconds);

    virtual void onEndAdvanceTimeStep(double timeStepInSeconds);

    void resolveCollision();

    void resolveCollision(
        ArrayAccessor1<Vector3D> newPositions,
        ArrayAccessor1<Vector3D> newVelocities);

    void setParticleSystemData(const ParticleSystemData3Ptr& newParticles);

 private:
    double _dragCoefficient = 1e-4;
    double _restitutionCoefficient = 0.0;
    Vector3D _gravity = Vector3D(0.0, kGravity, 0.0);

    ParticleSystemData3Ptr _particleSystemData;
    ParticleSystemData3::VectorData _newPositions;
    ParticleSystemData3::VectorData _newVelocities;
    Collider3Ptr _collider;
    VectorField3Ptr _wind;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);

    void accumulateExternalForces();

    void timeIntegration(double timeStepInSeconds);
};

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_SYSTEM_SOLVER3_H_
