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
    class Builder;

    //! Default constructor.
    ColliderSet2();

    //! Constructs with other colliders.
    explicit ColliderSet2(const std::vector<Collider2Ptr>& others);

    //! Returns the velocity of the collider at given \p point.
    Vector2D velocityAt(const Vector2D& point) const override;

    //! Adds a collider to the set.
    void addCollider(const Collider2Ptr& collider);

    //! Returns builder fox ColliderSet2.
    static Builder builder();

 private:
    std::vector<Collider2Ptr> _colliders;
};

//! Shared pointer for the ColliderSet2 type.
typedef std::shared_ptr<ColliderSet2> ColliderSet2Ptr;


//!
//! \brief Front-end to create ColliderSet2 objects step by step.
//!
class ColliderSet2::Builder final {
 public:
    //! Returns builder with other colliders.
    Builder& withColliders(const std::vector<Collider2Ptr>& others);

    //! Builds ColliderSet2.
    ColliderSet2 build() const;

    //! Builds shared pointer of ColliderSet2 instance.
    ColliderSet2Ptr makeShared() const;

 private:
    std::vector<Collider2Ptr> _colliders;
};

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER_SET2_H_
