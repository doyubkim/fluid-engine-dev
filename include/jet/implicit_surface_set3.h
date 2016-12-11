// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_IMPLICIT_SURFACE_SET3_H_
#define INCLUDE_JET_IMPLICIT_SURFACE_SET3_H_

#include <jet/implicit_surface3.h>
#include <vector>

namespace jet {

//!
//! \brief 3-D implicit surface set.
//!
//! This class represents 3-D implicit surface set which extends
//! ImplicitSurface3 by overriding implicit surface-related quries. This is
//! class can hold a collection of other implicit surface instances.
//!
class ImplicitSurfaceSet3 final : public ImplicitSurface3 {
 public:
    class Builder;

    //! Constructs an empty implicit surface set.
    ImplicitSurfaceSet3();

    //! Constructs an implicit surface set using list of other surfaces.
    explicit ImplicitSurfaceSet3(
        const std::vector<ImplicitSurface3Ptr>& surfaces);

    //! Constructs an implicit surface set using list of other surfaces.
    explicit ImplicitSurfaceSet3(const std::vector<Surface3Ptr>& surfaces);

    //! Copy constructor.
    ImplicitSurfaceSet3(const ImplicitSurfaceSet3& other);

    //! Returns the number of implicit surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th implicit surface.
    const ImplicitSurface3Ptr& surfaceAt(size_t i) const;

    //! Adds an explicit surface instance.
    void addExplicitSurface(const Surface3Ptr& surface);

    //! Adds an implicit surface instance.
    void addSurface(const ImplicitSurface3Ptr& surface);

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

    // ImplicitSurface3 implementations

    //! Returns signed distance from the given point \p otherPoint.
    double signedDistance(const Vector3D& otherPoint) const override;

    //! Returns builder fox ImplicitSurfaceSet3.
    static Builder builder();

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;

 private:
    std::vector<ImplicitSurface3Ptr> _surfaces;
};

//! Shared pointer type for the ImplicitSurfaceSet3.
typedef std::shared_ptr<ImplicitSurfaceSet3> ImplicitSurfaceSet3Ptr;

//!
//! \brief Front-end to create ImplicitSurfaceSet3 objects step by step.
//!
class ImplicitSurfaceSet3::Builder final {
 public:
    //! Returns builder with surfaces.
    Builder& withSurfaces(const std::vector<ImplicitSurface3Ptr>& surfaces);

    //! Returns builder with explicit surfaces.
    Builder& withExplicitSurfaces(
        const std::vector<Surface3Ptr>& surfaces);

    //! Builds ImplicitSurfaceSet3.
    ImplicitSurfaceSet3 build() const;

    //! Builds shared pointer of ImplicitSurfaceSet3 instance.
    ImplicitSurfaceSet3Ptr makeShared() const {
        return std::make_shared<ImplicitSurfaceSet3>(_surfaces);
    }

 private:
    std::vector<ImplicitSurface3Ptr> _surfaces;
};

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_SET3_H_
