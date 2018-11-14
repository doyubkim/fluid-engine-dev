// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_LIST_QUERY_ENGINE_H_
#define INCLUDE_JET_LIST_QUERY_ENGINE_H_

#include <jet/array.h>
#include <jet/intersection_query_engine.h>
#include <jet/nearest_neighbor_query_engine.h>

namespace jet {

//! Ad-hoc list-based N-D intersection/nearest-neighbor query engine.
template <typename T, size_t N>
class ListQueryEngine final : public IntersectionQueryEngine<T, N>,
                              public NearestNeighborQueryEngine<T, N> {
 public:
    //! Adds an item to the container.
    void add(const T& item);

    //! Adds items to the container.
    void add(const ConstArrayView1<T>& items);

    //! Returns true if given \p box intersects with any of the stored items.
    bool intersects(
        const BoundingBox<double, N>& box,
        const BoxIntersectionTestFunc<T, N>& testFunc) const override;

    //! Returns true if given \p ray intersects with any of the stored items.
    bool intersects(
        const Ray<double, N>& ray,
        const RayIntersectionTestFunc<T, N>& testFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const BoundingBox<double, N>& box,
        const BoxIntersectionTestFunc<T, N>& testFunc,
        const IntersectionVisitorFunc<T>& visitorFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const Ray<double, N>& ray,
        const RayIntersectionTestFunc<T, N>& testFunc,
        const IntersectionVisitorFunc<T>& visitorFunc) const override;

    //! Returns the closest intersection for given \p ray.
    ClosestIntersectionQueryResult<T, N> closestIntersection(
        const Ray<double, N>& ray,
        const GetRayIntersectionFunc<T, N>& testFunc) const override;

    //! Returns the nearest neighbor for given point and distance measure
    //! function.
    NearestNeighborQueryResult<T, N> nearest(
        const Vector<double, N>& pt,
        const NearestNeighborDistanceFunc<T, N>& distanceFunc) const override;

 private:
    Array1<T> _items;
};

//! 2-D ListQueryEngine type.
template <typename T>
using ListQueryEngine2 = ListQueryEngine<T, 2>;

//! 3-D ListQueryEngine type.
template <typename T>
using ListQueryEngine3 = ListQueryEngine<T, 3>;

}  // namespace jet

#include "detail/list_query_engine-inl.h"

#endif  // INCLUDE_JET_LIST_QUERY_ENGINE2_H_
