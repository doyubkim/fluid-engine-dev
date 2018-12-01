// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_BOX_H_
#define INCLUDE_JET_BOX_H_

#include <jet/bounding_box.h>
#include <jet/surface.h>

namespace jet {

//!
//! \brief N-D box geometry.
//!
//! This class represents N-D box geometry which extends Surface class by
//! overriding surface-related queries. This box implementation is an
//! axis-aligned box that wraps lower-level primitive type, BoundingBox.
//!
template <size_t N>
class Box final : public Surface<N> {
 public:
    class Builder;

    //! Bounding box of this box.
    BoundingBox<double, N> bound = BoundingBox<double, N>(
        Vector<double, N>(), Vector<double, N>::makeConstant(1.0));

    //! Constructs (0, 0, ...) x (1, 1, ...) box.
    Box(const Transform<N>& transform = Transform<N>(),
        bool isNormalFlipped = false);

    //! Constructs a box with given \p lowerCorner and \p upperCorner.
    Box(const Vector<double, N>& lowerCorner,
        const Vector<double, N>& upperCorner,
        const Transform<N>& transform = Transform<N>(),
        bool isNormalFlipped = false);

    //! Constructs a box with BoundingBox instance.
    Box(const BoundingBox<double, N>& boundingBox,
        const Transform<N>& transform = Transform<N>(),
        bool isNormalFlipped = false);

    //! Copy constructor.
    Box(const Box& other);

    //! Returns builder fox Box.
    static Builder builder();

 protected:
    // Surface implementations

    Vector<double, N> closestPointLocal(
        const Vector<double, N>& otherPoint) const override;

    bool intersectsLocal(const Ray<double, N>& ray) const override;

    BoundingBox<double, N> boundingBoxLocal() const override;

    Vector<double, N> closestNormalLocal(
        const Vector<double, N>& otherPoint) const override;

    SurfaceRayIntersection<N> closestIntersectionLocal(
        const Ray<double, N>& ray) const override;
};

//! 2-D Box type.
using Box2 = Box<2>;

//! 3-D Box type.
using Box3 = Box<3>;

//! Shared pointer type for the Box2.
using Box2Ptr = std::shared_ptr<Box2>;

//! Shared pointer type for the Box3.
using Box3Ptr = std::shared_ptr<Box3>;

//!
//! \brief Front-end to create Box objects step by step.
//!
template <size_t N>
class Box<N>::Builder final : public SurfaceBuilderBase<N, typename Box<N>::Builder> {
 public:
    //! Returns builder with lower corner set.
    Builder& withLowerCorner(const Vector<double, N>& pt);

    //! Returns builder with upper corner set.
    Builder& withUpperCorner(const Vector<double, N>& pt);

    //! Returns builder with bounding box.
    Builder& withBoundingBox(const BoundingBox<double, N>& bbox);

    //! Builds Box.
    Box build() const;

    //! Builds shared pointer of Box instance.
    std::shared_ptr<Box<N>> makeShared() const;

 private:
    using Base = SurfaceBuilderBase<N, typename Box<N>::Builder>;
    using Base::_isNormalFlipped;
    using Base::_transform;

    Vector<double, N> _lowerCorner;
    Vector<double, N> _upperCorner = Vector<double, N>::makeConstant(1.0);
};

}  // namespace jet

#endif  // INCLUDE_JET_BOX_H_
