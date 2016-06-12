// Copyright (c) 2016 Doyub Kim

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
    //! Plane normal.
    Vector2D normal = Vector2D(0, 1);

    //! Point that lies on the plane.
    Vector2D point;

    //! Constructs a plane that crosses (0, 0) with surface normal (0, 1).
    Plane2();

    //! Constructs a plane that cross \p point with surface normal \p normal.
    Plane2(const Vector2D& normal, const Vector2D& point);

    //! Copy constructor.
    Plane2(const Plane2& other);

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector2D& otherPoint) const override;

    //!
    //! \brief Returns the closest surface normal from the given point
    //! \p otherPoint.
    //!
    //! This function returns the "actual" closest surface normal from the
    //! given point \p otherPoint, meaning that the return value is not flipped
    //! regardless how Surface2::isNormalFlipped is set.
    //!
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this plane object.
    bool intersects(const Ray2D& ray) const override;

    //! Returns the closest intersection point for given \p ray.
    SurfaceRayIntersection2 closestIntersection(
        const Ray2D& ray) const override;

    //! Returns the bounding box of this plane object.
    BoundingBox2D boundingBox() const override;
};

typedef std::shared_ptr<Plane2> Plane2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PLANE2_H_
