// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CYLINDER3_H_
#define INCLUDE_JET_CYLINDER3_H_

#include <jet/surface3.h>

namespace jet {

//!
//! \brief 3-D cylinder geometry.
//!
//! This class represents 3-D cylinder geometry which extends Surface3 by
//! overriding surface-related queries. The cylinder is aligned with the y-axis.
//!
class Cylinder3 final : public Surface3 {
 public:
    //! Center of the cylinder.
    Vector3D center;

    //! Radius of the cylinder.
    double radius = 1.0;

    //! Height of the cylinder.
    double height = 1.0;

    //! Constructs a cylinder with
    Cylinder3();

    //! Constructs a cylinder with \p center, \p radius, and \p height.
    Cylinder3(const Vector3D& center, double radius, double height);

    //! Copy constructor.
    Cylinder3(const Cylinder3& other);

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

    //! Returns true if the given \p ray intersects with this cylinder object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the closest intersection point for given \p ray.
    SurfaceRayIntersection3 closestIntersection(
        const Ray3D& ray) const override;

    //! Returns the bounding box of this cylinder object.
    BoundingBox3D boundingBox() const override;
};

typedef std::shared_ptr<Cylinder3> Cylinder3Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_CYLINDER3_H_
