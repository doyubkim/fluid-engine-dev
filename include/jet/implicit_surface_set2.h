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

    //! Copy constructor.
    ImplicitSurfaceSet2(const ImplicitSurfaceSet2& other);

    //! Returns the number of implicit surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th implicit surface.
    const ImplicitSurface2Ptr& surfaceAt(size_t i) const;

    //! Adds an explicit surface instance.
    void addExplicitSurface(const Surface2Ptr& surface);

    //! Adds an implicit surface instance.
    void addSurface(const ImplicitSurface2Ptr& surface);

    // Surface2 implementations

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
    std::vector<ImplicitSurface2Ptr> _surfaces;
};

typedef std::shared_ptr<ImplicitSurfaceSet2> ImplicitSurfaceSet2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_
