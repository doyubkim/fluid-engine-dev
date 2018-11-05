// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SURFACE_H_
#define INCLUDE_JET_SURFACE_H_

#include <jet/bounding_box.h>
#include <jet/constants.h>
#include <jet/ray.h>
#include <jet/transform.h>

#include <memory>

namespace jet {

//! Struct that represents ray-surface intersection point.
template <size_t N>
struct SurfaceRayIntersection {
    bool isIntersecting = false;
    double distance = kMaxD;
    Vector<double, N> point;
    Vector<double, N> normal;
};

using SurfaceRayIntersection2 = SurfaceRayIntersection<2>;
using SurfaceRayIntersection3 = SurfaceRayIntersection<3>;

//! Abstract base class for N-D surface.
template <size_t N>
class Surface {
 public:
    //! Local-to-world transform.
    Transform<N> transform;

    //! Flips normal.
    bool isNormalFlipped = false;

    //! Constructs a surface with normal direction.
    Surface(const Transform<N>& transform = Transform<N>(),
            bool isNormalFlipped = false);

    //! Copy constructor.
    Surface(const Surface& other);

    //! Default destructor.
    virtual ~Surface();

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector<double, N> closestPoint(const Vector<double, N>& otherPoint) const;

    //! Returns the bounding box of this surface object.
    BoundingBox<double, N> boundingBox() const;

    //! Returns true if the given \p ray intersects with this surface object.
    bool intersects(const Ray<double, N>& ray) const;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector<double, N>& otherPoint) const;

    //! Returns the closest intersection point for given \p ray.
    SurfaceRayIntersection<N> closestIntersection(
        const Ray<double, N>& ray) const;

    //! Returns the normal to the closest point on the surface from the given
    //! point \p otherPoint.
    Vector<double, N> closestNormal(const Vector<double, N>& otherPoint) const;

    //! Updates internal spatial query engine.
    virtual void updateQueryEngine();

    //! Returns true if bounding box can be defined.
    virtual bool isBounded() const;

    //! Returns true if the surface is a valid geometry.
    virtual bool isValidGeometry() const;

 protected:
    //! Returns the closest point from the given point \p otherPoint to the
    //! surface in local frame.
    virtual Vector<double, N> closestPointLocal(
        const Vector<double, N>& otherPoint) const = 0;

    //! Returns the bounding box of this surface object in local frame.
    virtual BoundingBox<double, N> boundingBoxLocal() const = 0;

    //! Returns the closest intersection point for given \p ray in local frame.
    virtual SurfaceRayIntersection<N> closestIntersectionLocal(
        const Ray<double, N>& ray) const = 0;

    //! Returns the normal to the closest point on the surface from the given
    //! point \p otherPoint in local frame.
    virtual Vector<double, N> closestNormalLocal(
        const Vector<double, N>& otherPoint) const = 0;

    //! Returns true if the given \p ray intersects with this surface object
    //! in local frame.
    virtual bool intersectsLocal(const Ray<double, N>& ray) const;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface in local frame.
    virtual double closestDistanceLocal(
        const Vector<double, N>& otherPoint) const;
};

//! 2-D Surface type.
using Surface2 = Surface<2>;

//! 3-D Surface type.
using Surface3 = Surface<3>;

//! Shared pointer for the Surface2 type.
using Surface2Ptr = std::shared_ptr<Surface2>;

//! Shared pointer for the Surface3 type.
using Surface3Ptr = std::shared_ptr<Surface3>;

//!
//! \brief Base class for N-D surface builder.
//!
template <size_t N, typename DerivedBuilder>
class SurfaceBuilderBase {
 public:
    //! Returns builder with flipped normal flag.
    DerivedBuilder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with translation.
    DerivedBuilder& withTranslation(const Vector<double, N>& translation);

    //! Returns builder with orientation.
    DerivedBuilder& withOrientation(const Orientation<N>& orientation);

    //! Returns builder with transform.
    DerivedBuilder& withTransform(const Transform<N>& transform);

 protected:
    bool _isNormalFlipped = false;
    Transform<N> _transform;
};

template <size_t N, typename T>
T& SurfaceBuilderBase<N, T>::withIsNormalFlipped(bool isNormalFlipped) {
    _isNormalFlipped = isNormalFlipped;
    return static_cast<T&>(*this);
}

template <size_t N, typename T>
T& SurfaceBuilderBase<N, T>::withTranslation(
    const Vector<double, N>& translation) {
    _transform.setTranslation(translation);
    return static_cast<T&>(*this);
}

template <size_t N, typename T>
T& SurfaceBuilderBase<N, T>::withOrientation(
    const Orientation<N>& orientation) {
    _transform.setOrientation(orientation);
    return static_cast<T&>(*this);
}

template <size_t N, typename T>
T& SurfaceBuilderBase<N, T>::withTransform(const Transform<N>& transform) {
    _transform = transform;
    return static_cast<T&>(*this);
}

template <typename T>
using SurfaceBuilderBase2 = SurfaceBuilderBase<2, T>;

template <typename T>
using SurfaceBuilderBase3 = SurfaceBuilderBase<3, T>;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_H_
