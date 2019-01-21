// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_
#define INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_

#include <jet/bvh2.h>
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
    class Builder;

    //! Constructs an empty implicit surface set.
    ImplicitSurfaceSet2();

    //! Constructs an implicit surface set using list of other surfaces.
    ImplicitSurfaceSet2(const std::vector<ImplicitSurface2Ptr>& surfaces,
                        const Transform2& transform = Transform2(),
                        bool isNormalFlipped = false);

    //! Constructs an implicit surface set using list of other surfaces.
    ImplicitSurfaceSet2(const std::vector<Surface2Ptr>& surfaces,
                        const Transform2& transform = Transform2(),
                        bool isNormalFlipped = false);

    //! Copy constructor.
    ImplicitSurfaceSet2(const ImplicitSurfaceSet2& other);

    //! Updates internal spatial query engine.
    void updateQueryEngine() override;

    //! Returns true if bounding box can be defined.
    bool isBounded() const override;

    //! Returns true if the surface is a valid geometry.
    bool isValidGeometry() const override;

    //! Returns the number of implicit surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th implicit surface.
    const ImplicitSurface2Ptr& surfaceAt(size_t i) const;

    //! Adds an explicit surface instance.
    void addExplicitSurface(const Surface2Ptr& surface);

    //! Adds an implicit surface instance.
    void addSurface(const ImplicitSurface2Ptr& surface);

    //! Returns builder fox ImplicitSurfaceSet2.
    static Builder builder();

 private:
    std::vector<ImplicitSurface2Ptr> _surfaces;
    std::vector<ImplicitSurface2Ptr> _unboundedSurfaces;
    mutable Bvh2<ImplicitSurface2Ptr> _bvh;
    mutable bool _bvhInvalidated = true;

    // Surface2 implementations

    Vector2D closestPointLocal(const Vector2D& otherPoint) const override;

    BoundingBox2D boundingBoxLocal() const override;

    double closestDistanceLocal(const Vector2D& otherPoint) const override;

    bool intersectsLocal(const Ray2D& ray) const override;

    Vector2D closestNormalLocal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 closestIntersectionLocal(
        const Ray2D& ray) const override;

    bool isInsideLocal(const Vector2D& otherPoint) const override;

    // ImplicitSurface2 implementations

    double signedDistanceLocal(const Vector2D& otherPoint) const override;

    void invalidateBvh();

    void buildBvh() const;
};

//! Shared pointer type for the ImplicitSurfaceSet2.
typedef std::shared_ptr<ImplicitSurfaceSet2> ImplicitSurfaceSet2Ptr;

//!
//! \brief Front-end to create ImplicitSurfaceSet2 objects step by step.
//!
class ImplicitSurfaceSet2::Builder final
    : public SurfaceBuilderBase2<ImplicitSurfaceSet2::Builder> {
 public:
    //! Returns builder with surfaces.
    Builder& withSurfaces(const std::vector<ImplicitSurface2Ptr>& surfaces);

    //! Returns builder with explicit surfaces.
    Builder& withExplicitSurfaces(const std::vector<Surface2Ptr>& surfaces);

    //! Builds ImplicitSurfaceSet2.
    ImplicitSurfaceSet2 build() const;

    //! Builds shared pointer of ImplicitSurfaceSet2 instance.
    ImplicitSurfaceSet2Ptr makeShared() const;

 private:
    bool _isNormalFlipped = false;
    std::vector<ImplicitSurface2Ptr> _surfaces;
};

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_
