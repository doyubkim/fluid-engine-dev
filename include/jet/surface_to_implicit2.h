// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_
#define INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_

#include <jet/implicit_surface2.h>
#include <memory>

namespace jet {

//!
//! \brief 2-D implicit surface wrapper for generic Surface2 instance.
//!
//! This class represents 2-D implicit surface that converts Surface2 instance
//! to an ImplicitSurface2 object.
//!
class SurfaceToImplicit2 final : public ImplicitSurface2 {
 public:
    //! Constructs an instance with generic Surface2 instance.
    explicit SurfaceToImplicit2(const Surface2Ptr& surface);

    //! Copy constructor.
    SurfaceToImplicit2(const SurfaceToImplicit2& other);

    //! Returns the raw surface instance.
    Surface2Ptr surface() const;

    // Surface2 implementation

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector2D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this object.
    bool intersects(const Ray2D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox2D boundingBox() const override;

    // ImplicitSurface2 implementations

    //! Returns signed distance from the given point \p otherPoint.
    double signedDistance(const Vector2D& otherPoint) const override;

 protected:
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 actualClosestIntersection(
        const Ray2D& ray) const override;

 private:
    Surface2Ptr _surface;
};

//! Shared pointer for the SurfaceToImplicit2 type.
typedef std::shared_ptr<SurfaceToImplicit2> SurfaceToImplicit2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_
