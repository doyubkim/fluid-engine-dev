// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_LIST_QUERY_ENGINE2_INL_H_
#define INCLUDE_JET_DETAIL_LIST_QUERY_ENGINE2_INL_H_

#include <jet/list_query_engine2.h>

namespace jet {

template <typename T>
void ListQueryEngine2<T>::add(const T& item) {
    _items.push_back(item);
}

template <typename T>
void ListQueryEngine2<T>::add(const std::vector<T>& items) {
    _items.insert(_items.end(), items.begin(), items.end());
}

template <typename T>
bool ListQueryEngine2<T>::intersects(
    const BoundingBox2D& box,
    const BoxIntersectionTestFunc2<T>& testFunc) const {
    for (const auto& item : _items) {
        if (testFunc(item, box)) {
            return true;
        }
    }

    return false;
}

template <typename T>
bool ListQueryEngine2<T>::intersects(
    const Ray2D& ray, const RayIntersectionTestFunc2<T>& testFunc) const {
    for (const auto& item : _items) {
        if (testFunc(item, ray)) {
            return true;
        }
    }

    return false;
}

template <typename T>
void ListQueryEngine2<T>::forEachIntersectingItem(
    const BoundingBox2D& box, const BoxIntersectionTestFunc2<T>& testFunc,
    const IntersectionVisitorFunc2<T>& visitorFunc) const {
    for (const auto& item : _items) {
        if (testFunc(item, box)) {
            visitorFunc(item);
        }
    }
}

template <typename T>
void ListQueryEngine2<T>::forEachIntersectingItem(
    const Ray2D& ray, const RayIntersectionTestFunc2<T>& testFunc,
    const IntersectionVisitorFunc2<T>& visitorFunc) const {
    for (const auto& item : _items) {
        if (testFunc(item, ray)) {
            visitorFunc(item);
        }
    }
}

template <typename T>
ClosestIntersectionQueryResult2<T> ListQueryEngine2<T>::closestIntersection(
    const Ray2D& ray, const GetRayIntersectionFunc2<T>& testFunc) const {
    ClosestIntersectionQueryResult2<T> best;
    for (const auto& item : _items) {
        double dist = testFunc(item, ray);
        if (dist < best.distance) {
            best.distance = dist;
            best.item = &item;
        }
    }

    return best;
}

template <typename T>
NearestNeighborQueryResult2<T> ListQueryEngine2<T>::nearest(
    const Vector2D& pt,
    const NearestNeighborDistanceFunc2<T>& distanceFunc) const {
    NearestNeighborQueryResult2<T> best;
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

#endif  // INCLUDE_JET_DETAIL_LIST_QUERY_ENGINE2_INL_H_
