// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_BOX2_H_
#define INCLUDE_JET_BOX2_H_

#include <jet/surface2.h>
#include <jet/bounding_box2.h>

namespace jet {

//!
//! \brief 2-D box geometry.
//!
//! This class represents 2-D box geometry which extends Surface2 by overriding
//! surface-related queries. This box implementation is an axis-aligned box
//! that wraps lower-level primitive type, BoundingBox2D.
//!
class Box2 final : public Surface2 {
 public:
    class Builder;

    //! Bounding box of this box.
    BoundingBox2D bound = BoundingBox2D(Vector2D(), Vector2D(1.0, 1.0));

    //! Constructs (0, 0) x (1, 1) box.
    explicit Box2(bool isNormalFlipped = false);

    //! Constructs a box with given \p lowerCorner and \p upperCorner.
    Box2(
        const Vector2D& lowerCorner,
        const Vector2D& upperCorner,
        bool isNormalFlipped = false);

    //! Constructs a box with BoundingBox2D instance.
    explicit Box2(
        const BoundingBox2D& boundingBox,
        bool isNormalFlipped = false);

    //! Copy constructor.
    Box2(const Box2& other);

    // Surface2 implementations

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector2D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this box object.
    bool intersects(const Ray2D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox2D boundingBox() const override;

    //! Returns builder fox Box2.
    static Builder builder();

 protected:
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 actualClosestIntersection(
        const Ray2D& ray) const override;
};

//! Shared pointer type for the Box2.
typedef std::shared_ptr<Box2> Box2Ptr;


//!
//! \brief Front-end to create Box2 objects step by step.
//!
class Box2::Builder final : public SurfaceBuilderBase2<Box2::Builder> {
 public:
    //! Returns builder with lower corner set.
    Builder& withLowerCorner(const Vector2D& pt);

    //! Returns builder with upper corner set.
    Builder& withUpperCorner(const Vector2D& pt);

    //! Returns builder with bounding box.
    Builder& withBoundingBox(const BoundingBox2D& bbox);

    //! Builds Box2.
    Box2 build() const;

    //! Builds shared pointer of Box2 instance.
    Box2Ptr makeShared() const;

 private:
    Vector2D _lowerCorner{0, 0};
    Vector2D _upperCorner{1, 1};
};

}  // namespace jet


#endif  // INCLUDE_JET_BOX2_H_
