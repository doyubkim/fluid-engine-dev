// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_BOX3_H_
#define INCLUDE_JET_BOX3_H_

#include <jet/surface3.h>
#include <jet/bounding_box3.h>

namespace jet {

//!
//! \brief 3-D box geometry.
//!
//! This class represents 3-D box geometry which extends Surface3 by overriding
//! surface-related queries. This box implementation is an axis-aligned box
//! that wraps lower-level primitive type, BoundingBox3D.
//!
class Box3 final : public Surface3 {
 public:
    //! Bounding box of this box.
    BoundingBox3D bound
        = BoundingBox3D(Vector3D(), Vector3D(1.0, 1.0, 1.0));

    //! Constructs (0, 0, 0) x (1, 1, 1) box.
    Box3();

    //! Constructs a box with given \p lowerCorner and \p upperCorner.
    Box3(const Vector3D& lowerCorner, const Vector3D& upperCorner);

    //! Constructs a box with BoundingBox3D instance.
    explicit Box3(const BoundingBox3D& boundingBox);

    //! Copy constructor.
    Box3(const Box3& other);

    // Surface3 implementations

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector3D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this box object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox3D boundingBox() const override;

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;
};

//! Shared pointer type for the Box3.
typedef std::shared_ptr<Box3> Box3Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_BOX3_H_
