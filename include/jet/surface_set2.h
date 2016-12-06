// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_SET2_H_
#define INCLUDE_JET_SURFACE_SET2_H_

#include <jet/surface2.h>
#include <vector>

namespace jet {

//!
//! \brief 2-D surface set.
//!
//! This class represents 2-D surface set which extends Surface2 by overriding
//! surface-related quries. This is class can hold a collection of other surface
//! instances.
//!
class SurfaceSet2 final : public Surface2 {
 public:
    //! Constructs an empty surface set.
    SurfaceSet2();

    //! Copy constructor.
    SurfaceSet2(const SurfaceSet2& other);

    //! Returns the number of surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th surface.
    const Surface2Ptr& surfaceAt(size_t i) const;

    //! Adds a surface instance.
    void addSurface(const Surface2Ptr& surface);

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

 protected:
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 actualClosestIntersection(
        const Ray2D& ray) const override;

 private:
    std::vector<Surface2Ptr> _surfaces;
};

//! Shared pointer for the SurfaceSet2 type.
typedef std::shared_ptr<SurfaceSet2> SurfaceSet2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_SET2_H_
