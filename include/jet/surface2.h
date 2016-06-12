// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE2_H_
#define INCLUDE_JET_SURFACE2_H_

#include <jet/bounding_box2.h>
#include <jet/constants.h>
#include <jet/geometry2.h>
#include <jet/ray2.h>

namespace jet {

//! Struct that represents ray-surface intersection point.
struct SurfaceRayIntersection2 {
    bool isIntersecting = false;
    double t = kMaxD;
    Vector2D point;
    Vector2D normal;
};

//! Abstract base class for 2-D surface.
class Surface2 : public Geometry2 {
 public:
    //! Flips normal when calling Surface2::closestNormal(...).
    bool isNormalFlipped = false;

    //! Default constructor.
    Surface2();

    //! Copy constructor.
    Surface2(const Surface2& other);

    //! Default destructor.
    virtual ~Surface2();

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    virtual Vector2D closestPoint(const Vector2D& otherPoint) const = 0;

    //! Returns the closest intersection point for given \p ray.
    virtual SurfaceRayIntersection2 closestIntersection(
        const Ray2D& ray) const = 0;

    //! Returns the bounding box of this surface object.
    virtual BoundingBox2D boundingBox() const = 0;

    //!
    //! \brief Returns the closest surface normal from the given point
    //! \p otherPoint.
    //!
    //! This function returns the "actual" closest surface normal from the
    //! given point \p otherPoint, meaning that the return value is not flipped
    //! regardless how Surface2::isNormalFlipped is set.
    //!
    virtual Vector2D actualClosestNormal(const Vector2D& otherPoint) const = 0;

    //! Returns true if the given \p ray intersects with this surface object.
    virtual bool intersects(const Ray2D& ray) const;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    virtual double closestDistance(const Vector2D& otherPoint) const;

    //! Returns the normal to the closest point on the surface from the given
    //! point \p otherPoint.
    Vector2D closestNormal(const Vector2D& otherPoint) const;
};

typedef std::shared_ptr<Surface2> Surface2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE2_H_
