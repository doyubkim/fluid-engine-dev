// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_
#define INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_

#include <jet/implicit_surface3.h>
#include <memory>

namespace jet {

//!
//! \brief 3-D implicit surface wrapper for generic Surface3 instance.
//!
//! This class represents 3-D implicit surface that converts Surface3 instance
//! to an ImplicitSurface3 object.
//!
class SurfaceToImplicit3 final : public ImplicitSurface3 {
 public:
    //! Constructs an instance with generic Surface3 instance.
    explicit SurfaceToImplicit3(const Surface3Ptr& surface);

    //! Copy constructor.
    SurfaceToImplicit3(const SurfaceToImplicit3& other);

    // Surface3 implementation

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

    //! Returns true if the given \p ray intersects with this object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the closest intersection point for given \p ray.
    SurfaceRayIntersection3 closestIntersection(
        const Ray3D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox3D boundingBox() const override;

    // ImplicitSurface3 implementations

    //! Returns signed distance from the given point \p otherPoint.
    double signedDistance(const Vector3D& otherPoint) const override;

 private:
    Surface3Ptr _surface;
};

typedef std::shared_ptr<SurfaceToImplicit3> SurfaceToImplicit3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_
