// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_COLLIDER_SET_H_
#define INCLUDE_JET_COLLIDER_SET_H_

#include <jet/array.h>
#include <jet/collider.h>
#include <jet/surface_set.h>

namespace jet {

//! Collection of N-D colliders
template <size_t N>
class ColliderSet final : public Collider<N> {
 public:
    class Builder;

    using Collider<N>::surface;
    using Collider<N>::setSurface;

    //! Default constructor.
    ColliderSet();

    //! Constructs with other colliders.
    explicit ColliderSet(
        const ConstArrayView1<std::shared_ptr<Collider<N>>>& others);

    //! Returns the velocity of the collider at given \p point.
    Vector<double, N> velocityAt(const Vector<double, N>& point) const override;

    //! Adds a collider to the set.
    void addCollider(const std::shared_ptr<Collider<N>>& collider);

    //! Returns number of colliders.
    size_t numberOfColliders() const;

    //! Returns collider at index \p i.
    std::shared_ptr<Collider<N>> collider(size_t i) const;

    //! Returns builder fox ColliderSet.
    static Builder builder();

 private:
    Array1<std::shared_ptr<Collider<N>>> _colliders;
};

//! 2-D ColliderSet type.
using ColliderSet2 = ColliderSet<2>;

//! 3-D ColliderSet type.
using ColliderSet3 = ColliderSet<3>;

//! Shared pointer for the ColliderSet2 type.
using ColliderSet2Ptr = std::shared_ptr<ColliderSet2>;

//! Shared pointer for the ColliderSet3 type.
using ColliderSet3Ptr = std::shared_ptr<ColliderSet3>;

//!
//! \brief Front-end to create ColliderSet objects step by step.
//!
template <size_t N>
class ColliderSet<N>::Builder final {
 public:
    //! Returns builder with other colliders.
    Builder& withColliders(
        const ConstArrayView1<std::shared_ptr<Collider<N>>>& others);

    //! Builds ColliderSet.
    ColliderSet build() const;

    //! Builds shared pointer of ColliderSet instance.
    std::shared_ptr<ColliderSet<N>> makeShared() const;

 private:
    Array1<std::shared_ptr<Collider<N>>> _colliders;
};

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER_SET_H_
