// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PLANE_H_
#define INCLUDE_JET_PLANE_H_

#include <jet/surface.h>

namespace jet {

//!
//! \brief N-D plane geometry.
//!
//! This class represents N-D plane geometry which extends Surface by
//! overriding surface-related queries.
//!
template <size_t N>
class Plane final : public Surface<N> {
 public:
    class Builder;

    //! Plane normal.
    Vector<double, N> normal = Vector<double, N>::makeUnitY();

    //! Point that lies on the plane.
    Vector<double, N> point;

    //! Constructs a plane that crosses (0, 0, ...) with surface normal
    //! (y-axis).
    Plane(const Transform<N>& transform = Transform<N>(),
          bool isNormalFlipped = false);

    //! Constructs a plane that cross \p point with surface normal \p normal.
    Plane(const Vector<double, N>& normal, const Vector<double, N>& point,
          const Transform<N>& transform = Transform<N>(),
          bool isNormalFlipped = false);

    //! Copy constructor.
    Plane(const Plane& other);

    //! Returns true if bounding box can be defined.
    bool isBounded() const override;

    //! Returns builder fox Plane.
    static Builder builder();

 private:
    Vector<double, N> closestPointLocal(
        const Vector<double, N>& otherPoint) const override;

    bool intersectsLocal(const Ray<double, N>& ray) const override;

    BoundingBox<double, N> boundingBoxLocal() const override;

    Vector<double, N> closestNormalLocal(
        const Vector<double, N>& otherPoint) const override;

    SurfaceRayIntersection<N> closestIntersectionLocal(
        const Ray<double, N>& ray) const override;
};

//! 2-D plane type.
using Plane2 = Plane<2>;

//! 3-D plane type.
using Plane3 = Plane<3>;

//! Shared pointer for the Plane2 type.
using Plane2Ptr = std::shared_ptr<Plane2>;

//! Shared pointer for the Plane3 type.
using Plane3Ptr = std::shared_ptr<Plane3>;

//!
//! \brief Front-end to create Plane objects step by step.
//!
template <size_t N>
class Plane<N>::Builder final : public SurfaceBuilderBase<N, Plane<N>::Builder> {
 public:
    //! Returns builder with plane normal.
    Builder& withNormal(const Vector<double, N>& normal);

    //! Returns builder with point on the plane.
    Builder& withPoint(const Vector<double, N>& point);

    //! Builds Plane.
    Plane build() const;

    //! Builds shared pointer of Plane instance.
    std::shared_ptr<Plane<N>> makeShared() const;

 private:
    using Base = SurfaceBuilderBase<N, Plane<N>::Builder>;
    using Base::_transform;
    using Base::_isNormalFlipped;

    Vector<double, N> _normal = Vector<double, N>::makeUnitY();
    Vector<double, N> _point;
};

}  // namespace jet

#endif  // INCLUDE_JET_PLANE_H_
