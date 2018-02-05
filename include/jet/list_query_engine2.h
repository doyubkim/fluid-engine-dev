// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_LIST_QUERY_ENGINE2_H_
#define INCLUDE_JET_LIST_QUERY_ENGINE2_H_

#include <jet/intersection_query_engine2.h>
#include <jet/nearest_neighbor_query_engine2.h>
#include <vector>

namespace jet {

//! Ad-hoc list-based 2-D intersection/nearest-neighbor query engine.
template <typename T>
class ListQueryEngine2 final : public IntersectionQueryEngine2<T>,
                               public NearestNeighborQueryEngine2<T> {
 public:
    //! Adds an item to the container.
    void add(const T& item);

    //! Adds items to the container.
    void add(const std::vector<T>& items);

    //! Returns true if given \p box intersects with any of the stored items.
    bool intersects(const BoundingBox2D& box,
                    const BoxIntersectionTestFunc2<T>& testFunc) const override;

    //! Returns true if given \p ray intersects with any of the stored items.
    bool intersects(const Ray2D& ray,
                    const RayIntersectionTestFunc2<T>& testFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const BoundingBox2D& box, const BoxIntersectionTestFunc2<T>& testFunc,
        const IntersectionVisitorFunc2<T>& visitorFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const Ray2D& ray, const RayIntersectionTestFunc2<T>& testFunc,
        const IntersectionVisitorFunc2<T>& visitorFunc) const override;

    //! Returns the closest intersection for given \p ray.
    ClosestIntersectionQueryResult2<T> closestIntersection(
        const Ray2D& ray,
        const GetRayIntersectionFunc2<T>& testFunc) const override;

    //! Returns the nearest neighbor for given point and distance measure
    //! function.
    NearestNeighborQueryResult2<T> nearest(
        const Vector2D& pt,
        const NearestNeighborDistanceFunc2<T>& distanceFunc) const override;

 private:
    std::vector<T> _items;
};

}  // namespace jet

#include "detail/list_query_engine2-inl.h"

#endif  // INCLUDE_JET_LIST_QUERY_ENGINE2_H_
