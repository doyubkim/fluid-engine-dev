// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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
    class Builder;

    //! Linear velocity of the rigid body.
    Vector2D linearVelocity;

    //! Angular velocity of the rigid body.
    double angularVelocity = 0.0;

    //! Constructs a collider with a surface.
    explicit RigidBodyCollider2(const Surface2Ptr& surface);

    //! Constructs a collider with a surface and other parameters.
    RigidBodyCollider2(
        const Surface2Ptr& surface,
        const Vector2D& linearVelocity,
        double angularVelocity);

    //! Returns the velocity of the collider at given \p point.
    Vector2D velocityAt(const Vector2D& point) const override;

    //! Returns builder fox RigidBodyCollider2.
    static Builder builder();
};

//! Shared pointer for the RigidBodyCollider2 type.
typedef std::shared_ptr<RigidBodyCollider2> RigidBodyCollider2Ptr;


//!
//! \brief Front-end to create RigidBodyCollider2 objects step by step.
//!
class RigidBodyCollider2::Builder final {
 public:
    //! Returns builder with surface.
    Builder& withSurface(const Surface2Ptr& surface);

    //! Returns builder with linear velocity.
    Builder& withLinearVelocity(const Vector2D& linearVelocity);

    //! Returns builder with angular velocity.
    Builder& withAngularVelocity(double angularVelocity);

    //! Builds RigidBodyCollider2.
    RigidBodyCollider2 build() const;

    //! Builds shared pointer of RigidBodyCollider2 instance.
    RigidBodyCollider2Ptr makeShared() const;

 private:
    Surface2Ptr _surface;
    Vector2D _linearVelocity{0, 0};
    double _angularVelocity = 0.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_RIGID_BODY_COLLIDER2_H_
