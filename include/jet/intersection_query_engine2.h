// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_INTERSECTION_QUERY_ENGINE2_H_
#define INCLUDE_JET_INTERSECTION_QUERY_ENGINE2_H_

#include <jet/bounding_box2.h>
#include <jet/constants.h>

#include <functional>

namespace jet {

//! Closest intersection query result.
template <typename T>
struct ClosestIntersectionQueryResult2 {
    const T* item = nullptr;
    double distance = kMaxD;
};

//! Closest intersection distance measure function.
template <typename T>
using ClosestIntersectionDistanceFunc2 =
    std::function<double(const T&, const Vector2D&)>;

//! Box-item intersection test function.
template <typename T>
using BoxIntersectionTestFunc2 =
    std::function<bool(const T&, const BoundingBox2D&)>;

//! Ray-item intersection test function.
template <typename T>
using RayIntersectionTestFunc2 = std::function<bool(const T&, const Ray2D&)>;

//! Ray-item closest intersection evaluation function.
template <typename T>
using GetRayIntersectionFunc2 = std::function<double(const T&, const Ray2D&)>;

//! Visitor function which is invoked for each intersecting item.
template <typename T>
using IntersectionVisitorFunc2 = std::function<void(const T&)>;

//! Abstract base class for 2-D intersection test query engine.
template <typename T>
class IntersectionQueryEngine2 {
 public:
    //! Returns true if given \p box intersects with any of the stored items.
    virtual bool intersects(
        const BoundingBox2D& box,
        const BoxIntersectionTestFunc2<T>& testFunc) const = 0;

    //! Returns true if given \p ray intersects with any of the stored items.
    virtual bool intersects(
        const Ray2D& ray,
        const RayIntersectionTestFunc2<T>& testFunc) const = 0;

    //! Invokes \p visitorFunc for every intersecting items.
    virtual void forEachIntersectingItem(
        const BoundingBox2D& box, const BoxIntersectionTestFunc2<T>& testFunc,
        const IntersectionVisitorFunc2<T>& visitorFunc) const = 0;

    //! Invokes \p visitorFunc for every intersecting items.
    virtual void forEachIntersectingItem(
        const Ray2D& ray, const RayIntersectionTestFunc2<T>& testFunc,
        const IntersectionVisitorFunc2<T>& visitorFunc) const = 0;

    //! Returns the closest intersection for given \p ray.
    virtual ClosestIntersectionQueryResult2<T> closestIntersection(
        const Ray2D& ray, const GetRayIntersectionFunc2<T>& testFunc) const = 0;
};

}  // namespace jet

#endif  // INCLUDE_JET_INTERSECTION_QUERY_ENGINE2_H_
