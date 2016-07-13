// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_COLLIDER_SET3_H_
#define INCLUDE_JET_COLLIDER_SET3_H_

#include <jet/collider3.h>
#include <jet/surface_set3.h>
#include <vector>

namespace jet {

//! Collection of 3-D colliders
class ColliderSet3 final : public Collider3 {
 public:
    //! Default constructor.
    ColliderSet3();

    //! Returns the velocity of the collider at given \p point.
    Vector3D velocityAt(const Vector3D& point) const override;

    //! Adds a collider to the set.
    void addCollider(const Collider3Ptr& collider);

 private:
    std::vector<Collider3Ptr> _colliders;
};

typedef std::shared_ptr<ColliderSet3> ColliderSet3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER_SET3_H_
