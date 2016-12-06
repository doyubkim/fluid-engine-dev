// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_SET3_H_
#define INCLUDE_JET_SURFACE_SET3_H_

#include <jet/surface3.h>
#include <vector>

namespace jet {

//!
//! \brief 3-D surface set.
//!
//! This class represents 3-D surface set which extends Surface3 by overriding
//! surface-related quries. This is class can hold a collection of other surface
//! instances.
//!
class SurfaceSet3 final : public Surface3 {
 public:
    //! Constructs an empty surface set.
    SurfaceSet3();

    //! Copy constructor.
    SurfaceSet3(const SurfaceSet3& other);

    //! Returns the number of surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th surface.
    const Surface3Ptr& surfaceAt(size_t i) const;

    //! Adds a surface instance.
    void addSurface(const Surface3Ptr& surface);

    // Surface3 implementations

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector3D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox3D boundingBox() const override;

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;

 private:
    std::vector<Surface3Ptr> _surfaces;
};

//! Shared pointer for the SurfaceSet2 type.
typedef std::shared_ptr<SurfaceSet3> SurfaceSet3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_SET3_H_
