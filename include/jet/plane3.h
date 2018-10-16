// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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
    Plane3(
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Constructs a plane that cross \p point with surface normal \p normal.
    Plane3(
        const Vector3D& normal,
        const Vector3D& point,
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Constructs a plane with three points on the surface. The normal will be
    //! set using the counter clockwise direction.
    Plane3(
        const Vector3D& point0,
        const Vector3D& point1,
        const Vector3D& point2,
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Copy constructor.
    Plane3(const Plane3& other);

    //! Returns true if bounding box can be defined.
    bool isBounded() const override;

    //! Returns builder fox Plane3.
    static Builder builder();

 protected:
    Vector3D closestPointLocal(const Vector3D& otherPoint) const override;

    bool intersectsLocal(const Ray3D& ray) const override;

    BoundingBox3D boundingBoxLocal() const override;

    Vector3D closestNormalLocal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 closestIntersectionLocal(
        const Ray3D& ray) const override;
};

//! Shared pointer for the Plane3 type.
typedef std::shared_ptr<Plane3> Plane3Ptr;


//!
//! \brief Front-end to create Plane3 objects step by step.
//!
class Plane3::Builder final : public SurfaceBuilderBase3<Plane3::Builder> {
 public:
    //! Returns builder with plane normal.
    Builder& withNormal(const Vector3D& normal);

    //! Returns builder with point on the plane.
    Builder& withPoint(const Vector3D& point);

    //! Builds Plane3.
    Plane3 build() const;

    //! Builds shared pointer of Plane3 instance.
    Plane3Ptr makeShared() const;

 private:
    Vector3D _normal{0, 1, 0};
    Vector3D _point{0, 0, 0};
};

}  // namespace jet

#endif  // INCLUDE_JET_PLANE3_H_
