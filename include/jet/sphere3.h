// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SPHERE3_H_
#define INCLUDE_JET_SPHERE3_H_

#include <jet/surface3.h>
#include <jet/bounding_box3.h>

namespace jet {

//!
//! \brief 3-D sphere geometry.
//!
//! This class represents 3-D sphere geometry which extends Surface3 by
//! overriding surface-related queries.
//!
class Sphere3 final : public Surface3 {
 public:
    class Builder;

    //! Center of the sphere.
    Vector3D center;

    //! Radius of the sphere.
    double radius = 1.0;

    //! Constructs a sphere with center at (0, 0, 0) and radius of 1.
    Sphere3(
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Constructs a sphere with \p center and \p radius.
    Sphere3(
        const Vector3D& center,
        double radius,
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Copy constructor.
    Sphere3(const Sphere3& other);

    //! Returns builder fox Sphere3.
    static Builder builder();

 private:
    Vector3D closestPointLocal(const Vector3D& otherPoint) const override;

    double closestDistanceLocal(const Vector3D& otherPoint) const override;

    bool intersectsLocal(const Ray3D& ray) const override;

    BoundingBox3D boundingBoxLocal() const override;

    Vector3D closestNormalLocal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 closestIntersectionLocal(
        const Ray3D& ray) const override;
};

//! Shared pointer for the Sphere3 type.
typedef std::shared_ptr<Sphere3> Sphere3Ptr;

//!
//! \brief Front-end to create Sphere3 objects step by step.
//!
class Sphere3::Builder final : public SurfaceBuilderBase3<Sphere3::Builder>{
 public:
    //! Returns builder with sphere center.
    Builder& withCenter(const Vector3D& center);

    //! Returns builder with sphere radius.
    Builder& withRadius(double radius);

    //! Builds Sphere3.
    Sphere3 build() const;

    //! Builds shared pointer of Sphere3 instance.
    Sphere3Ptr makeShared() const;

 private:
    Vector3D _center{0, 0, 0};
    double _radius = 0.0;
};

}  // namespace jet


#endif  // INCLUDE_JET_SPHERE3_H_
