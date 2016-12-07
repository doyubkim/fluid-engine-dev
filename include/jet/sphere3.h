// Copyright (c) 2016 Doyub Kim

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
    explicit Sphere3(bool isNormalFlipped = false);

    //! Constructs a sphere with \p center and \p radius.
    Sphere3(
        const Vector3D& center,
        double radius,
        bool isNormalFlipped = false);

    //! Copy constructor.
    Sphere3(const Sphere3& other);

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector3D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this sphere object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the bounding box of this sphere object.
    BoundingBox3D boundingBox() const override;

    //! Returns builder fox Sphere3.
    static Builder builder();

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    //! Note, the book has different name and interface. This function used to
    //! be getClosestIntersection, but now it is simply
    //! actualClosestIntersection. Also, the book's function do not return
    //! SurfaceRayIntersection3 instance, but rather takes a pointer to existing
    //! SurfaceRayIntersection3 instance and modify its contents.
    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;
};

//! Shared pointer for the Sphere3 type.
typedef std::shared_ptr<Sphere3> Sphere3Ptr;

//!
//! \brief Front-end to create Sphere3 objects step by step.
//!
class Sphere3::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with sphere center.
    Builder& withCenter(const Vector3D& center);

    //! Returns builder with sphere radius.
    Builder& withRadius(double radius);

    //! Builds Sphere3.
    Sphere3 build() const;

 private:
    bool _isNormalFlipped = false;
    Vector3D _center{0, 0, 0};
    double _radius = 0.0;
};

}  // namespace jet


#endif  // INCLUDE_JET_SPHERE3_H_
