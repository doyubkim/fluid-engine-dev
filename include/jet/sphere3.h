// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SPHERE3_H_
#define INCLUDE_JET_SPHERE3_H_

#include <jet/surface3.h>
#include <jet/bounding_box3.h>

namespace jet {

//!
//! \brief 3-D sphere geometry.
//!
//! This class represents 3-D sphere geometry which extends Surface3 by
//! overriding surface-related queries.
//!
class Sphere3 final : public Surface3 {
 public:
    //! Center of the sphere.
    Vector3D center;

    //! Radius of the sphere.
    double radius = 1.0;

    //! Constructs a sphere with center at (0, 0, 0) and radius of 1.
    Sphere3();

    //! Constructs a sphere with \p center and \p radius.
    Sphere3(const Vector3D& center, double radius);

    //! Copy constructor.
    Sphere3(const Sphere3& other);

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector3D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this sphere object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the bounding box of this sphere object.
    BoundingBox3D boundingBox() const override;

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;
};

typedef std::shared_ptr<Sphere3> Sphere3Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_SPHERE3_H_
