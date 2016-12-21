// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PLANE3_H_
#define INCLUDE_JET_PLANE3_H_

#include <jet/surface3.h>

namespace jet {

//!
//! \brief 3-D plane geometry.
//!
//! This class represents 3-D plane geometry which extends Surface3 by
//! overriding surface-related queries.
//!
class Plane3 final : public Surface3 {
 public:
    class Builder;

    //! Plane normal.
    Vector3D normal = Vector3D(0, 1, 0);

    //! Point that lies on the plane.
    Vector3D point;

    //! Constructs a plane that crosses (0, 0, 0) with surface normal (0, 1, 0).
    explicit Plane3(bool isNormalFlipped = false);

    //! Constructs a plane that cross \p point with surface normal \p normal.
    Plane3(
        const Vector3D& normal,
        const Vector3D& point,
        bool isNormalFlipped = false);

    //! Constructs a plane with three points on the surface. The normal will be
    //! set using the counter clockwise direction.
    Plane3(
        const Vector3D& point0,
        const Vector3D& point1,
        const Vector3D& point2,
        bool isNormalFlipped = false);

    //! Copy constructor.
    Plane3(const Plane3& other);

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector3D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this plane object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the bounding box of this plane object.
    BoundingBox3D boundingBox() const override;

    //! Returns builder fox Plane3.
    static Builder builder();

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;
};

//! Shared pointer for the Plane3 type.
typedef std::shared_ptr<Plane3> Plane3Ptr;


//!
//! \brief Front-end to create Plane3 objects step by step.
//!
class Plane3::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with plane normal.
    Builder& withNormal(const Vector3D& normal);

    //! Returns builder with point on the plane.
    Builder& withPoint(const Vector3D& point);

    //! Builds Plane3.
    Plane3 build() const;

    //! Builds shared pointer of Plane3 instance.
    Plane3Ptr makeShared() const;

 private:
    bool _isNormalFlipped = false;
    Vector3D _normal{0, 1, 0};
    Vector3D _point{0, 0, 0};
};

}  // namespace jet

#endif  // INCLUDE_JET_PLANE3_H_
