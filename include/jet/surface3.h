// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SURFACE3_H_
#define INCLUDE_JET_SURFACE3_H_

#include <jet/bounding_box3.h>
#include <jet/constants.h>
#include <jet/ray3.h>
#include <jet/transform3.h>
#include <memory>

namespace jet {

//! Struct that represents ray-surface intersection point.
struct SurfaceRayIntersection3 {
    bool isIntersecting = false;
    double distance = kMaxD;
    Vector3D point;
    Vector3D normal;
};

//! Abstract base class for 3-D surface.
class Surface3 {
 public:
    //! Local-to-world transform.
    Transform3 transform;

    //! Flips normal when calling Surface3::closestNormal(...).
    bool isNormalFlipped = false;

    //! Constructs a surface with normal direction.
    Surface3(const Transform3& transform = Transform3(),
             bool isNormalFlipped = false);

    //! Copy constructor.
    Surface3(const Surface3& other);

    //! Default destructor.
    virtual ~Surface3();

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const;

    //! Returns the bounding box of this surface object.
    BoundingBox3D boundingBox() const;

    //! Returns true if the given \p ray intersects with this surface object.
    bool intersects(const Ray3D& ray) const;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector3D& otherPoint) const;

    //! Returns the closest intersection point for given \p ray.
    SurfaceRayIntersection3 closestIntersection(const Ray3D& ray) const;

    //! Returns the normal to the closest point on the surface from the given
    //! point \p otherPoint.
    Vector3D closestNormal(const Vector3D& otherPoint) const;

    //! Updates internal spatial query engine.
    virtual void updateQueryEngine();

    //! Returns true if bounding box can be defined.
    virtual bool isBounded() const;

    //! Returns true if the surface is a valid geometry.
    virtual bool isValidGeometry() const;

    //! Returns true if \p otherPoint is inside the volume defined by the
    //! surface.
    bool isInside(const Vector3D& otherPoint) const;

 protected:
    //! Returns the closest point from the given point \p otherPoint to the
    //! surface in local frame.
    virtual Vector3D closestPointLocal(const Vector3D& otherPoint) const = 0;

    //! Returns the bounding box of this surface object in local frame.
    virtual BoundingBox3D boundingBoxLocal() const = 0;

    //! Returns the closest intersection point for given \p ray in local frame.
    virtual SurfaceRayIntersection3 closestIntersectionLocal(
        const Ray3D& ray) const = 0;

    //! Returns the normal to the closest point on the surface from the given
    //! point \p otherPoint in local frame.
    virtual Vector3D closestNormalLocal(const Vector3D& otherPoint) const = 0;

    //! Returns true if the given \p ray intersects with this surface object
    //! in local frame.
    virtual bool intersectsLocal(const Ray3D& ray) const;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface in local frame.
    virtual double closestDistanceLocal(const Vector3D& otherPoint) const;

    //! Returns true if \p otherPoint is inside by given \p depth the volume
    //! defined by the surface in local frame.
    virtual bool isInsideLocal(const Vector3D& otherPoint) const;
};

//! Shared pointer for the Surface3 type.
typedef std::shared_ptr<Surface3> Surface3Ptr;

//!
//! \brief Base class for 3-D surface builder.
//!
template <typename DerivedBuilder>
class SurfaceBuilderBase3 {
 public:
    //! Returns builder with flipped normal flag.
    DerivedBuilder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with translation.
    DerivedBuilder& withTranslation(const Vector3D& translation);

    //! Returns builder with orientation.
    DerivedBuilder& withOrientation(const QuaternionD& orientation);

    //! Returns builder with transform.
    DerivedBuilder& withTransform(const Transform3& transform);

 protected:
    bool _isNormalFlipped = false;
    Transform3 _transform;
};

template <typename T>
T& SurfaceBuilderBase3<T>::withIsNormalFlipped(bool isNormalFlipped) {
    _isNormalFlipped = isNormalFlipped;
    return static_cast<T&>(*this);
}

template <typename T>
T& SurfaceBuilderBase3<T>::withTranslation(const Vector3D& translation) {
    _transform.setTranslation(translation);
    return static_cast<T&>(*this);
}

template <typename T>
T& SurfaceBuilderBase3<T>::withOrientation(const QuaternionD& orientation) {
    _transform.setOrientation(orientation);
    return static_cast<T&>(*this);
}

template <typename T>
T& SurfaceBuilderBase3<T>::withTransform(const Transform3& transform) {
    _transform = transform;
    return static_cast<T&>(*this);
}

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE3_H_
