// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SURFACE_SET2_H_
#define INCLUDE_JET_SURFACE_SET2_H_

#include <jet/bvh2.h>
#include <jet/surface2.h>

#include <vector>

namespace jet {

//!
//! \brief 2-D surface set.
//!
//! This class represents 2-D surface set which extends Surface2 by overriding
//! surface-related queries. This is class can hold a collection of other
//! surface instances.
//!
class SurfaceSet2 final : public Surface2 {
 public:
    class Builder;

    //! Constructs an empty surface set.
    SurfaceSet2();

    //! Constructs with a list of other surfaces.
    explicit SurfaceSet2(const std::vector<Surface2Ptr>& others,
                         const Transform2& transform = Transform2(),
                         bool isNormalFlipped = false);

    //! Copy constructor.
    SurfaceSet2(const SurfaceSet2& other);

    //! Returns the number of surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th surface.
    const Surface2Ptr& surfaceAt(size_t i) const;

    //! Adds a surface instance.
    void addSurface(const Surface2Ptr& surface);

    //! Returns builder for SurfaceSet2.
    static Builder builder();

 private:
    std::vector<Surface2Ptr> _surfaces;
    mutable Bvh2<Surface2Ptr> _bvh;
    mutable bool _bvhInvalidated = true;

    // Surface2 implementations

    Vector2D closestPointLocal(const Vector2D& otherPoint) const override;

    BoundingBox2D boundingBoxLocal() const override;

    double closestDistanceLocal(const Vector2D& otherPoint) const override;

    bool intersectsLocal(const Ray2D& ray) const override;

    Vector2D closestNormalLocal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 closestIntersectionLocal(
        const Ray2D& ray) const override;

    void invalidateBvh();

    void buildBvh() const;
};

//! Shared pointer for the SurfaceSet2 type.
typedef std::shared_ptr<SurfaceSet2> SurfaceSet2Ptr;

//!
//! \brief Front-end to create SurfaceSet2 objects step by step.
//!
class SurfaceSet2::Builder final : public SurfaceBuilderBase2<SurfaceSet2> {
 public:
    //! Returns builder with other surfaces.
    Builder& withSurfaces(const std::vector<Surface2Ptr>& others);

    //! Builds SurfaceSet2.
    SurfaceSet2 build() const;

    //! Builds shared pointer of SurfaceSet2 instance.
    SurfaceSet2Ptr makeShared() const;

 private:
    std::vector<Surface2Ptr> _surfaces;
};

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_SET2_H_
