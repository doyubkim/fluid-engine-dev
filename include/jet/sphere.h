// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SPHERE_H_
#define INCLUDE_JET_SPHERE_H_

#include <jet/bounding_box.h>
#include <jet/surface.h>

namespace jet {

//!
//! \brief N-D sphere geometry.
//!
//! This class represents N-D sphere geometry which extends Surface2 by
//! overriding surface-related queries.
//!
template <size_t N>
class Sphere final : public Surface<N> {
 public:
    class Builder;

    //! Center of the sphere.
    Vector<double, N> center;

    //! Radius of the sphere.
    double radius = 1.0;

    //! Constructs a sphere with center at the origin and radius of 1.
    Sphere(const Transform<N>& transform = Transform<N>(),
           bool isNormalFlipped = false);

    //! Constructs a sphere with \p center and \p radius.
    Sphere(const Vector<double, N>& center, double radius,
           const Transform<N>& transform = Transform<N>(),
           bool isNormalFlipped = false);

    //! Copy constructor.
    Sphere(const Sphere& other);

    //! Returns builder fox Sphere.
    static Builder builder();

 private:
    Vector<double, N> closestPointLocal(
        const Vector<double, N>& otherPoint) const override;

    double closestDistanceLocal(
        const Vector<double, N>& otherPoint) const override;

    bool intersectsLocal(const Ray<double, N>& ray) const override;

    BoundingBox<double, N> boundingBoxLocal() const override;

    Vector<double, N> closestNormalLocal(
        const Vector<double, N>& otherPoint) const override;

    SurfaceRayIntersection<N> closestIntersectionLocal(
        const Ray<double, N>& ray) const override;
};

//! 2-D Sphere type.
using Sphere2 = Sphere<2>;

//! 3-D Sphere type.
using Sphere3 = Sphere<3>;

//! Shared pointer for the Sphere2 type.
using Sphere2Ptr = std::shared_ptr<Sphere2>;

//! Shared pointer for the Sphere3 type.
using Sphere3Ptr = std::shared_ptr<Sphere3>;

//!
//! \brief Front-end to create Sphere objects step by step.
//!
template <size_t N>
class Sphere<N>::Builder final
    : public SurfaceBuilderBase<N, Sphere<N>::Builder> {
 public:
    //! Returns builder with sphere center.
    Builder& withCenter(const Vector<double, N>& center);

    //! Returns builder with sphere radius.
    Builder& withRadius(double radius);

    //! Builds Sphere.
    Sphere<N> build() const;

    //! Builds shared pointer of Sphere instance.
    std::shared_ptr<Sphere<N>> makeShared() const;

 private:
    using Base = SurfaceBuilderBase<N, Sphere<N>::Builder>;
    using Base::_isNormalFlipped;
    using Base::_transform;

    Vector<double, N> _center;
    double _radius = 0.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_SPHERE2_H_
