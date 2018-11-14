// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_RIGID_BODY_COLLIDER_H_
#define INCLUDE_JET_RIGID_BODY_COLLIDER_H_

#include <jet/collider.h>

namespace jet {

template <size_t N>
class AngularVelocity {};

template <>
class AngularVelocity<2> {
 public:
    double value = 0.0;

    Vector2D cross(const Vector2D& r) const {
        return value * Vector2D(-r.y, r.x);
    }
};

    template <>
class AngularVelocity<3> {
 public:
    Vector3D value;

    Vector3D cross(const Vector3D& r) const {
        return value.cross(r);
    }
};

//!
//! \brief N-D rigid body collider class.
//!
//! This class implements N-D rigid body collider. The collider can only take
//! rigid body motion with linear and rotational velocities.
//!
template <size_t N>
class RigidBodyCollider final : public Collider<N> {
 public:
    class Builder;

    using Collider<N>::surface;
    using Collider<N>::setSurface;

    //! Linear velocity of the rigid body.
    Vector<double, N> linearVelocity;

    //! Angular velocity of the rigid body.
    AngularVelocity<N> angularVelocity;

    //! Constructs a collider with a surface.
    explicit RigidBodyCollider(const std::shared_ptr<Surface<N>>& surface);

    //! Constructs a collider with a surface and other parameters.
    RigidBodyCollider(const std::shared_ptr<Surface<N>>& surface,
                      const Vector<double, N>& linearVelocity,
                      const AngularVelocity<N>& angularVelocity);

    //! Returns the velocity of the collider at given \p point.
    Vector<double, N> velocityAt(const Vector<double, N>& point) const override;

    //! Returns builder fox RigidBodyCollider.
    static Builder builder();
};

//! 2-D RigidBodyCollider type.
using RigidBodyCollider2 = RigidBodyCollider<2>;

//! 3-D RigidBodyCollider type.
using RigidBodyCollider3 = RigidBodyCollider<3>;

//! Shared pointer for the RigidBodyCollider2 type.
using RigidBodyCollider2Ptr = std::shared_ptr<RigidBodyCollider2>;

//! Shared pointer for the RigidBodyCollider3 type.
using RigidBodyCollider3Ptr = std::shared_ptr<RigidBodyCollider3>;

//!
//! \brief Front-end to create RigidBodyCollider objects step by step.
//!
template <size_t N>
class RigidBodyCollider<N>::Builder final {
 public:
    //! Returns builder with surface.
    Builder& withSurface(const std::shared_ptr<Surface<N>>& surface);

    //! Returns builder with linear velocity.
    Builder& withLinearVelocity(const Vector<double, N>& linearVelocity);

    //! Returns builder with angular velocity.
    Builder& withAngularVelocity(const AngularVelocity<N>& angularVelocity);

    //! Builds RigidBodyCollider.
    RigidBodyCollider build() const;

    //! Builds shared pointer of RigidBodyCollider instance.
    std::shared_ptr<RigidBodyCollider<N>> makeShared() const;

 private:
    std::shared_ptr<Surface<N>> _surface;
    Vector<double, N> _linearVelocity;
    AngularVelocity<N> _angularVelocity;
};

}  // namespace jet

#endif  // INCLUDE_JET_RIGID_BODY_COLLIDER_H_
