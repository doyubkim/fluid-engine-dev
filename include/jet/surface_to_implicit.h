// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SURFACE_TO_IMPLICIT_H_
#define INCLUDE_JET_SURFACE_TO_IMPLICIT_H_

#include <jet/implicit_surface.h>

namespace jet {

//!
//! \brief N-D implicit surface wrapper for generic Surface instance.
//!
//! This class represents N-D implicit surface that converts Surface instance
//! to an ImplicitSurface object. The conversion is made by evaluating closest
//! point and normal from a given point for the given (explicit) surface. Thus,
//! this conversion won't work for every single surfaces. Use this class only
//! for the basic primitives such as Sphere or Box.
//!
template <size_t N>
class SurfaceToImplicit final : public ImplicitSurface<N> {
 public:
    class Builder;

    using ImplicitSurface<N>::transform;
    using ImplicitSurface<N>::isNormalFlipped;

    //! Constructs an instance with generic Surface2 instance.
    SurfaceToImplicit(const std::shared_ptr<Surface<N>>& surface,
                      const Transform<N>& transform = Transform<N>(),
                      bool isNormalFlipped = false);

    //! Copy constructor.
    SurfaceToImplicit(const SurfaceToImplicit& other);

    //! Returns true if bounding box can be defined.
    bool isBounded() const override;

    //! Returns true if the surface is a valid geometry.
    bool isValidGeometry() const override;

    //! Returns the raw surface instance.
    std::shared_ptr<Surface<N>> surface() const;

    //! Returns builder fox SurfaceToImplicit.
    static Builder builder();

 protected:
    Vector<double, N> closestPointLocal(
        const Vector<double, N>& otherPoint) const override;

    double closestDistanceLocal(
        const Vector<double, N>& otherPoint) const override;

    bool intersectsLocal(const Ray<double, N>& ray) const override;

    BoundingBox<double, N> boundingBoxLocal() const override;

    Vector<double, N> closestNormalLocal(
        const Vector<double, N>& otherPoint) const override;

    double signedDistanceLocal(
        const Vector<double, N>& otherPoint) const override;

    SurfaceRayIntersection<N> closestIntersectionLocal(
        const Ray<double, N>& ray) const override;

 private:
    std::shared_ptr<Surface<N>> _surface;
};

//! 2-D SurfaceToImplicit type.
using SurfaceToImplicit2 = SurfaceToImplicit<2>;

//! 3-D SurfaceToImplicit type.
using SurfaceToImplicit3 = SurfaceToImplicit<3>;

//! Shared pointer for the SurfaceToImplicit2 type.
using SurfaceToImplicit2Ptr = std::shared_ptr<SurfaceToImplicit2>;

//! Shared pointer for the SurfaceToImplicit3 type.
using SurfaceToImplicit3Ptr = std::shared_ptr<SurfaceToImplicit3>;

//!
//! \brief Front-end to create SurfaceToImplicit objects step by step.
//!
template <size_t N>
class SurfaceToImplicit<N>::Builder final
    : public SurfaceBuilderBase<N, typename SurfaceToImplicit<N>::Builder> {
 public:
    //! Returns builder with surface.
    Builder& withSurface(const std::shared_ptr<Surface<N>>& surface);

    //! Builds SurfaceToImplicit.
    SurfaceToImplicit build() const;

    //! Builds shared pointer of SurfaceToImplicit instance.
    std::shared_ptr<SurfaceToImplicit> makeShared() const;

 private:
    using Base = SurfaceBuilderBase<N, typename SurfaceToImplicit<N>::Builder>;
    using Base::_transform;
    using Base::_isNormalFlipped;

    std::shared_ptr<Surface<N>> _surface;
};

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_TO_IMPLICIT_H_
