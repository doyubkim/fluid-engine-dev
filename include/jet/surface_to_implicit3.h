// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_
#define INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_

#include <jet/implicit_surface3.h>
#include <memory>

namespace jet {

//!
//! \brief 3-D implicit surface wrapper for generic Surface3 instance.
//!
//! This class represents 3-D implicit surface that converts Surface3 instance
//! to an ImplicitSurface3 object.
//!
class SurfaceToImplicit3 final : public ImplicitSurface3 {
 public:
    class Builder;

    //! Constructs an instance with generic Surface3 instance.
    explicit SurfaceToImplicit3(
        const Surface3Ptr& surface,
        bool isNormalFlipped = false);

    //! Copy constructor.
    SurfaceToImplicit3(const SurfaceToImplicit3& other);

    //! Returns the raw surface instance.
    Surface3Ptr surface() const;

    // Surface3 implementation

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

    //! Returns builder fox SurfaceToImplicit2.
    static Builder builder();

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;

 private:
    Surface3Ptr _surface;
};

//! Shared pointer for the SurfaceToImplicit3 type.
typedef std::shared_ptr<SurfaceToImplicit3> SurfaceToImplicit3Ptr;


//!
//! \brief Front-end to create SurfaceToImplicit3 objects step by step.
//!
class SurfaceToImplicit3::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with surface.
    Builder& withSurface(const Surface3Ptr& surface);

    //! Builds SurfaceToImplicit3.
    SurfaceToImplicit3 build() const;

    //! Builds shared pointer of SurfaceToImplicit3 instance.
    SurfaceToImplicit3Ptr makeShared() const {
        return std::make_shared<SurfaceToImplicit3>(
            _surface,
            _isNormalFlipped);
    }

 private:
    bool _isNormalFlipped = false;
    Surface3Ptr _surface;
};

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_
