// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_COLLIDER_SET2_H_
#define INCLUDE_JET_COLLIDER_SET2_H_

#include <jet/collider2.h>
#include <jet/surface_set2.h>
#include <vector>

namespace jet {

//! Collection of 2-D colliders
class ColliderSet2 final : public Collider2 {
 public:
    //! Default constructor.
    ColliderSet2();

    //! Returns the velocity of the collider at given \p point.
    Vector2D velocityAt(const Vector2D& point) const override;

    //! Adds a collider to the set.
    void addCollider(const Collider2Ptr& collider);

 private:
    std::vector<Collider2Ptr> _colliders;
};

//! Shared pointer for the ColliderSet2 type.
typedef std::shared_ptr<ColliderSet2> ColliderSet2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER_SET2_H_
