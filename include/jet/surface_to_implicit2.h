// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_
#define INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_

#include <jet/implicit_surface2.h>
#include <memory>

namespace jet {

//!
//! \brief 2-D implicit surface wrapper for generic Surface2 instance.
//!
//! This class represents 2-D implicit surface that converts Surface2 instance
//! to an ImplicitSurface2 object.
//!
class SurfaceToImplicit2 final : public ImplicitSurface2 {
 public:
    class Builder;

    //! Constructs an instance with generic Surface2 instance.
    SurfaceToImplicit2(
        const Surface2Ptr& surface,
        const Transform2& transform = Transform2(),
        bool isNormalFlipped = false);

    //! Copy constructor.
    SurfaceToImplicit2(const SurfaceToImplicit2& other);

    //! Returns the raw surface instance.
    Surface2Ptr surface() const;

    //! Returns builder fox SurfaceToImplicit2.
    static Builder builder();

 protected:
    Vector2D closestPointLocal(const Vector2D& otherPoint) const override;

    double closestDistanceLocal(const Vector2D& otherPoint) const override;

    bool intersectsLocal(const Ray2D& ray) const override;

    BoundingBox2D boundingBoxLocal() const override;

    Vector2D closestNormalLocal(
        const Vector2D& otherPoint) const override;

    double signedDistanceLocal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 closestIntersectionLocal(
        const Ray2D& ray) const override;

 private:
    Surface2Ptr _surface;
};

//! Shared pointer for the SurfaceToImplicit2 type.
typedef std::shared_ptr<SurfaceToImplicit2> SurfaceToImplicit2Ptr;


//!
//! \brief Front-end to create SurfaceToImplicit2 objects step by step.
//!
class SurfaceToImplicit2::Builder final
    : public SurfaceBuilderBase2<SurfaceToImplicit2::Builder> {
 public:
    //! Returns builder with surface.
    Builder& withSurface(const Surface2Ptr& surface);

    //! Builds SurfaceToImplicit2.
    SurfaceToImplicit2 build() const;

    //! Builds shared pointer of SurfaceToImplicit2 instance.
    SurfaceToImplicit2Ptr makeShared() const;

 private:
    Surface2Ptr _surface;
};

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_
