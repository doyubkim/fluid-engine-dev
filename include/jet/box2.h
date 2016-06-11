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
    //! Constructs (0, 0) x (1, 1) box.
    Box2();

    //! Constructs a box with given \p lowerCorner and \p upperCorner.
    Box2(const Vector2D& lowerCorner, const Vector2D& upperCorner);

    //! Constructs a box with BoundingBox2D instance.
    explicit Box2(const BoundingBox2D& boundingBox);

    //! Copy constructor.
    Box2(const Box2& other);

    // Surface2 implementations

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    //!
    //! \brief Returns the closest surface normal from the given point
    //! \p otherPoint.
    //!
    //! This function returns the "actual" closest surface normal from the
    //! given point \p otherPoint, meaning that the return value is not flipped
    //! regardless how Surface2::isNormalFlipped is set. For this class, the
    //! surface normal points outside the box.
    //!
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector2D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this box object.
    bool intersects(const Ray2D& ray) const override;

    //! Returns the closest intersection point for given \p ray.
    SurfaceRayIntersection2 closestIntersection(
        const Ray2D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox2D boundingBox() const override;

 private:
    BoundingBox2D _boundingBox
        = BoundingBox2D(Vector2D(), Vector2D(1.0, 1.0));
};

typedef std::shared_ptr<Box2> Box2Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_BOX2_H_
