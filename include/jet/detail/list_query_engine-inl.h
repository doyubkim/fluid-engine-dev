// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_LIST_QUERY_ENGINE_INL_H_
#define INCLUDE_JET_DETAIL_LIST_QUERY_ENGINE_INL_H_

#include <jet/list_query_engine.h>

namespace jet {

template <typename T, size_t N>
void ListQueryEngine<T, N>::add(const T& item) {
    _items.append(item);
}

template <typename T, size_t N>
void ListQueryEngine<T, N>::add(const ConstArrayView1<T>& items) {
    _items.append(items);
}

template <typename T, size_t N>
bool ListQueryEngine<T, N>::intersects(
    const BoundingBox<double, N>& box,
    const BoxIntersectionTestFunc<T, N>& testFunc) const {
    for (const auto& item : _items) {
        if (testFunc(item, box)) {
            return true;
        }
    }

    return false;
}

template <typename T, size_t N>
bool ListQueryEngine<T, N>::intersects(
    const Ray<double, N>& ray,
    const RayIntersectionTestFunc<T, N>& testFunc) const {
    for (const auto& item : _items) {
        if (testFunc(item, ray)) {
            return true;
        }
    }

    return false;
}

template <typename T, size_t N>
void ListQueryEngine<T, N>::forEachIntersectingItem(
    const BoundingBox<double, N>& box,
    const BoxIntersectionTestFunc<T, N>& testFunc,
    const IntersectionVisitorFunc<T>& visitorFunc) const {
    for (const auto& item : _items) {
        if (testFunc(item, box)) {
            visitorFunc(item);
        }
    }
}

template <typename T, size_t N>
void ListQueryEngine<T, N>::forEachIntersectingItem(
    const Ray<double, N>& ray, const RayIntersectionTestFunc<T, N>& testFunc,
    const IntersectionVisitorFunc<T>& visitorFunc) const {
    for (const auto& item : _items) {
        if (testFunc(item, ray)) {
            visitorFunc(item);
        }
    }
}

template <typename T, size_t N>
ClosestIntersectionQueryResult<T, N> ListQueryEngine<T, N>::closestIntersection(
    const Ray<double, N>& ray,
    const GetRayIntersectionFunc<T, N>& testFunc) const {
    ClosestIntersectionQueryResult<T, N> best;
    for (const auto& item : _items) {
        double dist = testFunc(item, ray);
        if (dist < best.distance) {
            best.distance = dist;
            best.item = &item;
        }
    }

    return best;
}

template <typename T, size_t N>
NearestNeighborQueryResult<T, N> ListQueryEngine<T, N>::nearest(
    const Vector<double, N>& pt,
    const NearestNeighborDistanceFunc<T, N>& distanceFunc) const {
    NearestNeighborQueryResult<T, N> best;
    for (const auto& item : _items) {
        double dist = distanceFunc(item, pt);
        if (dist < best.distance) {
            best.item = &item;
            best.distance = dist;
        }
    }

    return best;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_LIST_QUERY_ENGINE_INL_H_
