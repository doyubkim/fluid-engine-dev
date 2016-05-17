// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_RIGID_BODY_COLLIDER2_H_
#define INCLUDE_JET_RIGID_BODY_COLLIDER2_H_

#include <jet/collider2.h>

namespace jet {

class RigidBodyCollider2 final : public Collider2 {
 public:
    explicit RigidBodyCollider2(const Surface2Ptr& surface);

    Vector2D velocityAt(const Vector2D& point) const override;

    const Vector2D& linearVelocity() const;

    void setLinearVelocity(const Vector2D& newVelocity);

    const Vector2D& angularVelocity() const;

    void setAngularVelocity(const Vector2D& newVelocity);

    const Vector2D& origin() const;

    void setOrigin(const Vector2D& newOrigin);

 private:
    Vector2D _linearVelocity;
    Vector2D _angularVelocity;
    Vector2D _origin;
};

typedef std::shared_ptr<RigidBodyCollider2> RigidBodyCollider2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_RIGID_BODY_COLLIDER2_H_
