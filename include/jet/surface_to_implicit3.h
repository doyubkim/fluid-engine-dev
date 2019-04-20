// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_
#define INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_

#include <jet/implicit_surface3.h>
#include <memory>

namespace jet {

//!
//! \brief 3-D implicit surface wrapper for generic Surface3 instance.
//!
//! This class represents 3-D implicit surface that converts Surface3 instance
//! to an ImplicitSurface3 object. The conversion is made by evaluating closest
//! point and normal from a given point for the given (explicit) surface. Thus,
//! this conversion won't work for every single surfaces, especially
//! TriangleMesh3. To use TriangleMesh3 as an ImplicitSurface3 instance,
//! please take a look at ImplicitTriangleMesh3. Use this class only
//! for the basic primitives such as Sphere3 or Box3.
//!
class SurfaceToImplicit3 final : public ImplicitSurface3 {
 public:
    class Builder;

    //! Constructs an instance with generic Surface3 instance.
    SurfaceToImplicit3(const Surface3Ptr& surface,
                       const Transform3& transform = Transform3(),
                       bool isNormalFlipped = false);

    //! Copy constructor.
    SurfaceToImplicit3(const SurfaceToImplicit3& other);

    //! Updates internal spatial query engine.
    void updateQueryEngine() override;

    //! Returns true if bounding box can be defined.
    bool isBounded() const override;

    //! Returns true if the surface is a valid geometry.
    bool isValidGeometry() const override;

    //! Returns the raw surface instance.
    Surface3Ptr surface() const;

    //! Returns builder fox SurfaceToImplicit3.
    static Builder builder();

 protected:
    Vector3D closestPointLocal(const Vector3D& otherPoint) const override;

    double closestDistanceLocal(const Vector3D& otherPoint) const override;

    bool intersectsLocal(const Ray3D& ray) const override;

    BoundingBox3D boundingBoxLocal() const override;

    Vector3D closestNormalLocal(const Vector3D& otherPoint) const override;

    double signedDistanceLocal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 closestIntersectionLocal(
        const Ray3D& ray) const override;

    bool isInsideLocal(const Vector3D& otherPoint) const override;

 private:
    Surface3Ptr _surface;
};

//! Shared pointer for the SurfaceToImplicit3 type.
typedef std::shared_ptr<SurfaceToImplicit3> SurfaceToImplicit3Ptr;

//!
//! \brief Front-end to create SurfaceToImplicit3 objects step by step.
//!
class SurfaceToImplicit3::Builder final
    : public SurfaceBuilderBase3<SurfaceToImplicit3::Builder> {
 public:
    //! Returns builder with surface.
    Builder& withSurface(const Surface3Ptr& surface);

    //! Builds SurfaceToImplicit3.
    SurfaceToImplicit3 build() const;

    //! Builds shared pointer of SurfaceToImplicit3 instance.
    SurfaceToImplicit3Ptr makeShared() const;

 private:
    Surface3Ptr _surface;
};

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_
