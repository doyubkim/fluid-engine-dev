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
    class Builder;

    //! Bounding box of this box.
    BoundingBox3D bound
        = BoundingBox3D(Vector3D(), Vector3D(1.0, 1.0, 1.0));

    //! Constructs (0, 0, 0) x (1, 1, 1) box.
    explicit Box3(bool isNormalFlipped = false);

    //! Constructs a box with given \p lowerCorner and \p upperCorner.
    Box3(
        const Vector3D& lowerCorner,
        const Vector3D& upperCorner,
        bool isNormalFlipped = false);

    //! Constructs a box with BoundingBox3D instance.
    explicit Box3(
        const BoundingBox3D& boundingBox,
        bool isNormalFlipped = false);

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

    //! Returns builder fox Box3.
    static Builder builder();

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;
};

//! Shared pointer type for the Box3.
typedef std::shared_ptr<Box3> Box3Ptr;


//!
//! \brief Front-end to create Box3 objects step by step.
//!
class Box3::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with lower corner set.
    Builder& withLowerCorner(const Vector3D& pt);

    //! Returns builder with upper corner set.
    Builder& withUpperCorner(const Vector3D& pt);

    //! Returns builder with bounding box.
    Builder& withBoundingBox(const BoundingBox3D& bbox);

    //! Builds Box3.
    Box3 build() const;

    //! Builds shared pointer of Box3 instance.
    Box3Ptr makeShared() const;

 private:
    bool _isNormalFlipped = false;
    Vector3D _lowerCorner{0, 0, 0};
    Vector3D _upperCorner{1, 1, 1};
};

}  // namespace jet


#endif  // INCLUDE_JET_BOX3_H_
