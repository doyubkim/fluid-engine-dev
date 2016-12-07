// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PLANE2_H_
#define INCLUDE_JET_PLANE2_H_

#include <jet/surface2.h>

namespace jet {

//!
//! \brief 2-D plane geometry.
//!
//! This class represents 2-D plane geometry which extends Surface2 by
//! overriding surface-related queries.
//!
class Plane2 final : public Surface2 {
 public:
    class Builder;

    //! Plane normal.
    Vector2D normal = Vector2D(0, 1);

    //! Point that lies on the plane.
    Vector2D point;

    //! Constructs a plane that crosses (0, 0) with surface normal (0, 1).
    explicit Plane2(bool isNormalFlipped = false);

    //! Constructs a plane that cross \p point with surface normal \p normal.
    Plane2(
        const Vector2D& normal,
        const Vector2D& point,
        bool isNormalFlipped = false);

    //! Copy constructor.
    Plane2(const Plane2& other);

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector2D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this plane object.
    bool intersects(const Ray2D& ray) const override;


    //! Returns the bounding box of this plane object.
    BoundingBox2D boundingBox() const override;

    //! Returns builder fox Plane2.
    static Builder builder();

 protected:
    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 actualClosestIntersection(
        const Ray2D& ray) const override;
};

//! Shared pointer for the Plane2 type.
typedef std::shared_ptr<Plane2> Plane2Ptr;


//!
//! \brief Front-end to create Plane2 objects step by step.
//!
class Plane2::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with plane normal.
    Builder& withNormal(const Vector2D& normal);

    //! Returns builder with point on the plane.
    Builder& withPoint(const Vector2D& point);

    //! Builds Plane2.
    Plane2 build() const;

 private:
    bool _isNormalFlipped = false;
    Vector2D _normal{0, 1};
    Vector2D _point{0, 0};
};

}  // namespace jet

#endif  // INCLUDE_JET_PLANE2_H_
