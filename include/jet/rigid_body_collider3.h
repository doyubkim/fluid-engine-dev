// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_RIGID_BODY_COLLIDER3_H_
#define INCLUDE_JET_RIGID_BODY_COLLIDER3_H_

#include <jet/collider3.h>

namespace jet {

//!
//! \brief 3-D rigid body collider class.
//!
//! This class implements 3-D rigid body collider. The collider can only take
//! rigid body motion with linear and rotational velocities.
//!
class RigidBodyCollider3 final : public Collider3 {
 public:
    //! Linear velocity of the rigid body.
    Vector3D linearVelocity;

    //! Angular velocity of the rigid body.
    Vector3D angularVelocity;

    //! Origin of the rigid body rotation.
    Vector3D origin;

    //! Constructs a collider with a surface.
    explicit RigidBodyCollider3(const Surface3Ptr& surface);

    //! Returns the velocity of the collider at given \p point.
    Vector3D velocityAt(const Vector3D& point) const override;
};

typedef std::shared_ptr<RigidBodyCollider3> RigidBodyCollider3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_RIGID_BODY_COLLIDER3_H_
