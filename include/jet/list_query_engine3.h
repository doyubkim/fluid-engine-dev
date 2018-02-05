// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_LIST_QUERY_ENGINE3_H_
#define INCLUDE_JET_LIST_QUERY_ENGINE3_H_

#include <jet/intersection_query_engine3.h>
#include <jet/nearest_neighbor_query_engine3.h>
#include <vector>

namespace jet {

//! Ad-hoc list-based 3-D intersection/nearest-neighbor query engine.
template <typename T>
class ListQueryEngine3 final : public IntersectionQueryEngine3<T>,
                               public NearestNeighborQueryEngine3<T> {
 public:
    //! Adds an item to the container.
    void add(const T& item);

    //! Adds items to the container.
    void add(const std::vector<T>& items);

    //! Returns true if given \p box intersects with any of the stored items.
    bool intersects(const BoundingBox3D& box,
                    const BoxIntersectionTestFunc3<T>& testFunc) const override;

    //! Returns true if given \p ray intersects with any of the stored items.
    bool intersects(const Ray3D& ray,
                    const RayIntersectionTestFunc3<T>& testFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const BoundingBox3D& box, const BoxIntersectionTestFunc3<T>& testFunc,
        const IntersectionVisitorFunc3<T>& visitorFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const Ray3D& ray, const RayIntersectionTestFunc3<T>& testFunc,
        const IntersectionVisitorFunc3<T>& visitorFunc) const override;

    //! Returns the closest intersection for given \p ray.
    ClosestIntersectionQueryResult3<T> closestIntersection(
        const Ray3D& ray,
        const GetRayIntersectionFunc3<T>& testFunc) const override;

    //! Returns the nearest neighbor for given point and distance measure
    //! function.
    NearestNeighborQueryResult3<T> nearest(
        const Vector3D& pt,
        const NearestNeighborDistanceFunc3<T>& distanceFunc) const override;

 private:
    std::vector<T> _items;
};

}  // namespace jet

#include "detail/list_query_engine3-inl.h"

#endif  // INCLUDE_JET_LIST_QUERY_ENGINE3_H_
