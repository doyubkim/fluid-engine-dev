// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_COLLIDER_SET3_H_
#define INCLUDE_JET_COLLIDER_SET3_H_

#include <jet/collider3.h>
#include <jet/surface_set3.h>
#include <vector>

namespace jet {

//! Collection of 3-D colliders
class ColliderSet3 final : public Collider3 {
 public:
    class Builder;

    //! Default constructor.
    ColliderSet3();

    //! Constructs with other colliders.
    explicit ColliderSet3(const std::vector<Collider3Ptr>& others);

    //! Returns the velocity of the collider at given \p point.
    Vector3D velocityAt(const Vector3D& point) const override;

    //! Adds a collider to the set.
    void addCollider(const Collider3Ptr& collider);

    //! Returns number of colliders.
    size_t numberOfColliders() const;

    //! Returns collider at index \p i.
    Collider3Ptr collider(size_t i) const;

    //! Returns builder fox ColliderSet3.
    static Builder builder();

 private:
    std::vector<Collider3Ptr> _colliders;
};

//! Shared pointer for the ColliderSet3 type.
typedef std::shared_ptr<ColliderSet3> ColliderSet3Ptr;


//!
//! \brief Front-end to create ColliderSet3 objects step by step.
//!
class ColliderSet3::Builder final {
 public:
    //! Returns builder with other colliders.
    Builder& withColliders(const std::vector<Collider3Ptr>& others);

    //! Builds ColliderSet3.
    ColliderSet3 build() const;

    //! Builds shared pointer of ColliderSet3 instance.
    ColliderSet3Ptr makeShared() const;

 private:
    std::vector<Collider3Ptr> _colliders;
};

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER_SET3_H_
