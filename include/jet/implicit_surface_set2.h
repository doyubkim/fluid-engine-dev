// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_
#define INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_

#include <jet/implicit_surface2.h>
#include <vector>

namespace jet {

//!
//! \brief 2-D implicit surface set.
//!
//! This class represents 2-D implicit surface set which extends
//! ImplicitSurface2 by overriding implicit surface-related quries. This is
//! class can hold a collection of other implicit surface instances.
//!
class ImplicitSurfaceSet2 final : public ImplicitSurface2 {
 public:
    //! Constructs an empty implicit surface set.
    ImplicitSurfaceSet2();

    //! Returns the number of implicit surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th implicit surface.
    const ImplicitSurface2Ptr& surfaceAt(size_t i) const;

    //! Adds an implicit surface instance.
    void addSurface(const Surface2Ptr& surface);

    //! Adds an implicit surface instance.
    void addImplicitSurface(const ImplicitSurface2Ptr& surface);

    // Surface2 implementations

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    //!
    //! \brief Returns the closest surface normal from the given point
    //! \p otherPoint.
    //!
    //! This function returns the "actual" closest surface normal from the
    //! given point \p otherPoint, meaning that the return value is not flipped
    //! regardless how Surface2::isNormalFlipped is set.
    //!
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this object.
    bool intersects(const Ray2D& ray) const override;

    //! Returns the closest intersection point for given \p ray.
    SurfaceRayIntersection2 closestIntersection(
        const Ray2D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox2D boundingBox() const override;

    // ImplicitSurface2 implementations

    //! Returns signed distance from the given point \p otherPoint.
    double signedDistance(const Vector2D& otherPoint) const override;

 private:
    std::vector<ImplicitSurface2Ptr> _surfaces;
};

typedef std::shared_ptr<ImplicitSurfaceSet2> ImplicitSurfaceSet2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_
