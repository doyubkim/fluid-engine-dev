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
    ImplicitSurfaceSet3(
        const std::vector<ImplicitSurface3Ptr>& surfaces,
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Constructs an implicit surface set using list of other surfaces.
    ImplicitSurfaceSet3(
        const std::vector<Surface3Ptr>& surfaces,
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

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

    //! Returns builder fox ImplicitSurfaceSet3.
    static Builder builder();

 private:
    std::vector<ImplicitSurface3Ptr> _surfaces;

    // Surface3 implementations

    Vector3D closestPointLocal(const Vector3D& otherPoint) const override;

    BoundingBox3D boundingBoxLocal() const override;

    double closestDistanceLocal(const Vector3D& otherPoint) const override;

    bool intersectsLocal(const Ray3D& ray) const override;

    Vector3D closestNormalLocal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 closestIntersectionLocal(
        const Ray3D& ray) const override;

    // ImplicitSurface3 implementations

    double signedDistanceLocal(const Vector3D& otherPoint) const override;
};

//! Shared pointer type for the ImplicitSurfaceSet3.
typedef std::shared_ptr<ImplicitSurfaceSet3> ImplicitSurfaceSet3Ptr;


//!
//! \brief Front-end to create ImplicitSurfaceSet3 objects step by step.
//!
class ImplicitSurfaceSet3::Builder final
    : public SurfaceBuilderBase3<ImplicitSurfaceSet3::Builder> {
 public:
    //! Returns builder with surfaces.
    Builder& withSurfaces(const std::vector<ImplicitSurface3Ptr>& surfaces);

    //! Returns builder with explicit surfaces.
    Builder& withExplicitSurfaces(
        const std::vector<Surface3Ptr>& surfaces);

    //! Builds ImplicitSurfaceSet3.
    ImplicitSurfaceSet3 build() const;

    //! Builds shared pointer of ImplicitSurfaceSet3 instance.
    ImplicitSurfaceSet3Ptr makeShared() const;

 private:
    bool _isNormalFlipped = false;
    std::vector<ImplicitSurface3Ptr> _surfaces;
};

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_SET3_H_
