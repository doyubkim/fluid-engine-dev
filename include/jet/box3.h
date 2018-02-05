// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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
    Box3(
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Constructs a box with given \p lowerCorner and \p upperCorner.
    Box3(
        const Vector3D& lowerCorner,
        const Vector3D& upperCorner,
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Constructs a box with BoundingBox3D instance.
    explicit Box3(
        const BoundingBox3D& boundingBox,
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Copy constructor.
    Box3(const Box3& other);

    //! Returns builder fox Box3.
    static Builder builder();

 protected:
    // Surface3 implementations

    Vector3D closestPointLocal(const Vector3D& otherPoint) const override;

    bool intersectsLocal(const Ray3D& ray) const override;

    BoundingBox3D boundingBoxLocal() const override;

    Vector3D closestNormalLocal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 closestIntersectionLocal(
        const Ray3D& ray) const override;
};

//! Shared pointer type for the Box3.
typedef std::shared_ptr<Box3> Box3Ptr;


//!
//! \brief Front-end to create Box3 objects step by step.
//!
class Box3::Builder final : public SurfaceBuilderBase3<Box3::Builder> {
 public:
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
    Vector3D _lowerCorner{0, 0, 0};
    Vector3D _upperCorner{1, 1, 1};
};

}  // namespace jet


#endif  // INCLUDE_JET_BOX3_H_
