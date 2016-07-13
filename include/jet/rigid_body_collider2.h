// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_RIGID_BODY_COLLIDER2_H_
#define INCLUDE_JET_RIGID_BODY_COLLIDER2_H_

#include <jet/collider2.h>

namespace jet {

//!
//! \brief 2-D rigid body collider class.
//!
//! This class implements 2-D rigid body collider. The collider can only take
//! rigid body motion with linear and rotational velocities.
//!
class RigidBodyCollider2 final : public Collider2 {
 public:
    //! Linear velocity of the rigid body.
    Vector2D linearVelocity;

    //! Angular velocity of the rigid body.
    double angularVelocity;

    //! Origin of the rigid body rotation.
    Vector2D origin;

    //! Constructs a collider with a surface.
    explicit RigidBodyCollider2(const Surface2Ptr& surface);

    //! Returns the velocity of the collider at given \p point.
    Vector2D velocityAt(const Vector2D& point) const override;
};

typedef std::shared_ptr<RigidBodyCollider2> RigidBodyCollider2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_RIGID_BODY_COLLIDER2_H_
