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
    class Builder;

    //! Constructs an empty surface set.
    SurfaceSet2();

    //! Constructs with a list of other surfaces.
    explicit SurfaceSet2(
        const std::vector<Surface2Ptr>& others,
        bool isNormalFlipped = false);

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

    //! Returns builder for SurfaceSet2.
    static Builder builder();

 protected:
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 actualClosestIntersection(
        const Ray2D& ray) const override;

 private:
    std::vector<Surface2Ptr> _surfaces;
};

//! Shared pointer for the SurfaceSet2 type.
typedef std::shared_ptr<SurfaceSet2> SurfaceSet2Ptr;

//!
//! \brief Front-end to create SurfaceSet2 objects step by step.
//!
class SurfaceSet2::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with other surfaces.
    Builder& withSurfaces(const std::vector<Surface2Ptr>& others);

    //! Builds SurfaceSet2.
    SurfaceSet2 build() const;

    //! Builds shared pointer of SurfaceSet2 instance.
    SurfaceSet2Ptr makeShared() const;

 private:
    bool _isNormalFlipped = false;
    std::vector<Surface2Ptr> _surfaces;
};

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_SET2_H_
