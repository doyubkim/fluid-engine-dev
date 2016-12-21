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
    class Builder;

    //! Constructs an empty surface set.
    SurfaceSet3();

    //! Constructs with a list of other surfaces.
    explicit SurfaceSet3(
        const std::vector<Surface3Ptr>& others,
        bool isNormalFlipped = false);

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

    //! Returns builder for SurfaceSet3.
    static Builder builder();

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;

 private:
    std::vector<Surface3Ptr> _surfaces;
};

//! Shared pointer for the SurfaceSet2 type.
typedef std::shared_ptr<SurfaceSet3> SurfaceSet3Ptr;


//!
//! \brief Front-end to create SurfaceSet3 objects step by step.
//!
class SurfaceSet3::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with other surfaces.
    Builder& withSurfaces(const std::vector<Surface3Ptr>& others);

    //! Builds SurfaceSet3.
    SurfaceSet3 build() const;

    //! Builds shared pointer of SurfaceSet3 instance.
    SurfaceSet3Ptr makeShared() const;

 private:
    bool _isNormalFlipped = false;
    std::vector<Surface3Ptr> _surfaces;
};

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_SET3_H_
