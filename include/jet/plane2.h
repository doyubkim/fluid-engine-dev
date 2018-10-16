// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PLANE2_H_
#define INCLUDE_JET_PLANE2_H_

#include <jet/surface2.h>

namespace jet {

//!
//! \brief 2-D plane geometry.
//!
//! This class represents 2-D plane geometry which extends Surface2 by
//! overriding surface-related queries.
//!
class Plane2 final : public Surface2 {
 public:
    class Builder;

    //! Plane normal.
    Vector2D normal = Vector2D(0, 1);

    //! Point that lies on the plane.
    Vector2D point;

    //! Constructs a plane that crosses (0, 0) with surface normal (0, 1).
    Plane2(
        const Transform2& transform = Transform2(),
        bool isNormalFlipped = false);

    //! Constructs a plane that cross \p point with surface normal \p normal.
    Plane2(
        const Vector2D& normal,
        const Vector2D& point,
        const Transform2& transform = Transform2(),
        bool isNormalFlipped = false);

    //! Copy constructor.
    Plane2(const Plane2& other);

    //! Returns true if bounding box can be defined.
    bool isBounded() const override;

    //! Returns builder fox Plane2.
    static Builder builder();

 private:
    Vector2D closestPointLocal(const Vector2D& otherPoint) const override;

    bool intersectsLocal(const Ray2D& ray) const override;

    BoundingBox2D boundingBoxLocal() const override;

    Vector2D closestNormalLocal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 closestIntersectionLocal(
        const Ray2D& ray) const override;
};

//! Shared pointer for the Plane2 type.
typedef std::shared_ptr<Plane2> Plane2Ptr;


//!
//! \brief Front-end to create Plane2 objects step by step.
//!
class Plane2::Builder final : public SurfaceBuilderBase2<Plane2::Builder> {
 public:
    //! Returns builder with plane normal.
    Builder& withNormal(const Vector2D& normal);

    //! Returns builder with point on the plane.
    Builder& withPoint(const Vector2D& point);

    //! Builds Plane2.
    Plane2 build() const;

    //! Builds shared pointer of Plane2 instance.
    Plane2Ptr makeShared() const;

 private:
    Vector2D _normal{0, 1};
    Vector2D _point{0, 0};
};

}  // namespace jet

#endif  // INCLUDE_JET_PLANE2_H_
