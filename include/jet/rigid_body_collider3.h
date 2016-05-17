// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_RIGID_BODY_COLLIDER3_H_
#define INCLUDE_JET_RIGID_BODY_COLLIDER3_H_

#include <jet/collider3.h>

namespace jet {

class RigidBodyCollider3 final : public Collider3 {
 public:
    explicit RigidBodyCollider3(const Surface3Ptr& surface);

    Vector3D velocityAt(const Vector3D& point) const override;

    const Vector3D& linearVelocity() const;

    void setLinearVelocity(const Vector3D& newVelocity);

    const Vector3D& angularVelocity() const;

    void setAngularVelocity(const Vector3D& newVelocity);

    const Vector3D& origin() const;

    void setOrigin(const Vector3D& newOrigin);

 private:
    Vector3D _linearVelocity;
    Vector3D _angularVelocity;
    Vector3D _origin;
};

typedef std::shared_ptr<RigidBodyCollider3> RigidBodyCollider3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_RIGID_BODY_COLLIDER3_H_
