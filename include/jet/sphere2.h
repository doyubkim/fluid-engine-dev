// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SPHERE2_H_
#define INCLUDE_JET_SPHERE2_H_

#include <jet/surface2.h>
#include <jet/bounding_box2.h>

namespace jet {

//!
//! \brief 2-D sphere geometry.
//!
//! This class represents 2-D sphere geometry which extends Surface2 by
//! overriding surface-related queries.
//!
class Sphere2 final : public Surface2 {
 public:
    //! Center of the sphere.
    Vector2D center;

    //! Radius of the sphere.
    double radius = 1.0;

    //! Constructs a sphere with center at (0, 0) and radius of 1.
    Sphere2();

    //! Constructs a sphere with \p center and \p radius.
    Sphere2(const Vector2D& center, double radius);

    //! Copy constructor.
    Sphere2(const Sphere2& other);

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector2D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this plane object.
    bool intersects(const Ray2D& ray) const override;

    //! Returns the bounding box of this plane object.
    BoundingBox2D boundingBox() const override;

 protected:
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 actualClosestIntersection(
        const Ray2D& ray) const override;
};

//! Shared pointer for the Sphere2 type.
typedef std::shared_ptr<Sphere2> Sphere2Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_SPHERE2_H_
