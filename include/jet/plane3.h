// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PLANE3_H_
#define INCLUDE_JET_PLANE3_H_

#include <jet/surface3.h>

namespace jet {

//!
//! \brief 3-D plane geometry.
//!
//! This class represents 3-D plane geometry which extends Surface3 by
//! overriding surface-related queries.
//!
class Plane3 final : public Surface3 {
 public:
    //! Plane normal.
    Vector3D normal = Vector3D(0, 1, 0);

    //! Point that lies on the plane.
    Vector3D point;

    //! Constructs a plane that crosses (0, 0, 0) with surface normal (0, 1, 0).
    Plane3();

    //! Constructs a plane that cross \p point with surface normal \p normal.
    Plane3(const Vector3D& normal, const Vector3D& point);

    //! Constructs a plane with three points on the surface. The normal will be
    //! set using the counter clockwise direction.
    Plane3(
        const Vector3D& point0,
        const Vector3D& point1,
        const Vector3D& point2);

    //! Copy constructor.
    Plane3(const Plane3& other);

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector3D& otherPoint) const override;

    //!
    //! \brief Returns the closest surface normal from the given point
    //! \p otherPoint.
    //!
    //! This function returns the "actual" closest surface normal from the
    //! given point \p otherPoint, meaning that the return value is not flipped
    //! regardless how Surface3::isNormalFlipped is set.
    //!
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this plane object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the closest intersection point for given \p ray.
    SurfaceRayIntersection3 closestIntersection(
        const Ray3D& ray) const override;

    //! Returns the bounding box of this plane object.
    BoundingBox3D boundingBox() const override;
};

typedef std::shared_ptr<Plane3> Plane3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PLANE3_H_
