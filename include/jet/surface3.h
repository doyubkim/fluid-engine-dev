// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE3_H_
#define INCLUDE_JET_SURFACE3_H_

#include <jet/bounding_box3.h>
#include <jet/constants.h>
#include <jet/geometry3.h>
#include <jet/ray3.h>

namespace jet {

//! Struct that represents ray-surface intersection point.
struct SurfaceRayIntersection3 {
    bool isIntersecting = false;
    double t = kMaxD;
    Vector3D point;
    Vector3D normal;
};

//! Abstract base class for 3-D surface.
class Surface3 : public Geometry3 {
 public:
    //! Flips normal when calling Surface3::closestNormal(...).
    bool isNormalFlipped = false;

    //! Default constructor.
    Surface3();

    //! Copy constructor.
    Surface3(const Surface3& other);

    //! Default destructor.
    virtual ~Surface3();

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    virtual Vector3D closestPoint(const Vector3D& otherPoint) const = 0;

    //! Returns the closest intersection point for given \p ray.
    virtual SurfaceRayIntersection3 closestIntersection(
        const Ray3D& ray) const = 0;

    //! Returns the bounding box of this surface object.
    virtual BoundingBox3D boundingBox() const = 0;

    //!
    //! \brief Returns the closest surface normal from the given point
    //! \p otherPoint.
    //!
    //! This function returns the "actual" closest surface normal from the
    //! given point \p otherPoint, meaning that the return value is not flipped
    //! regardless how Surface3::isNormalFlipped is set.
    //!
    virtual Vector3D actualClosestNormal(const Vector3D& otherPoint) const = 0;

    //! Returns true if the given \p ray intersects with this surface object.
    virtual bool intersects(const Ray3D& ray) const;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    virtual double closestDistance(const Vector3D& otherPoint) const;

    //! Returns the normal to the closest point on the surface from the given
    //! point \p otherPoint.
    Vector3D closestNormal(const Vector3D& otherPoint) const;
};

typedef std::shared_ptr<Surface3> Surface3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE3_H_
