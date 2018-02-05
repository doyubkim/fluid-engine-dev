// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_BOUNDING_BOX2_INL_H_
#define INCLUDE_JET_DETAIL_BOUNDING_BOX2_INL_H_

#include <algorithm>
#include <limits>
#include <utility>  // just make cpplint happy..

namespace jet {

template <typename T>
BoundingBox<T, 2>::BoundingBox() {
    reset();
}

template <typename T>
BoundingBox<T, 2>::BoundingBox(const Vector2<T>& point1,
                               const Vector2<T>& point2) {
    lowerCorner.x = std::min(point1.x, point2.x);
    lowerCorner.y = std::min(point1.y, point2.y);
    upperCorner.x = std::max(point1.x, point2.x);
    upperCorner.y = std::max(point1.y, point2.y);
}

template <typename T>
BoundingBox<T, 2>::BoundingBox(const BoundingBox& other)
    : lowerCorner(other.lowerCorner), upperCorner(other.upperCorner) {}

template <typename T>
T BoundingBox<T, 2>::width() const {
    return upperCorner.x - lowerCorner.x;
}

template <typename T>
T BoundingBox<T, 2>::height() const {
    return upperCorner.y - lowerCorner.y;
}

template <typename T>
T BoundingBox<T, 2>::length(size_t axis) {
    return upperCorner[axis] - lowerCorner[axis];
}

template <typename T>
bool BoundingBox<T, 2>::overlaps(const BoundingBox& other) const {
    if (upperCorner.x < other.lowerCorner.x ||
        lowerCorner.x > other.upperCorner.x) {
        return false;
    }

    if (upperCorner.y < other.lowerCorner.y ||
        lowerCorner.y > other.upperCorner.y) {
        return false;
    }

    return true;
}

template <typename T>
bool BoundingBox<T, 2>::contains(const Vector2<T>& point) const {
    if (upperCorner.x < point.x || lowerCorner.x > point.x) {
        return false;
    }

    if (upperCorner.y < point.y || lowerCorner.y > point.y) {
        return false;
    }

    return true;
}

template <typename T>
bool BoundingBox<T, 2>::intersects(const Ray2<T>& ray) const {
    T tMin = 0;
    T tMax = std::numeric_limits<T>::max();

    const Vector2<T>& rayInvDir = ray.direction.rdiv(1);

    for (int i = 0; i < 2; ++i) {
        T tNear = (lowerCorner[i] - ray.origin[i]) * rayInvDir[i];
        T tFar = (upperCorner[i] - ray.origin[i]) * rayInvDir[i];

        if (tNear > tFar) {
            std::swap(tNear, tFar);
        }

        tMin = std::max(tNear, tMin);
        tMax = std::min(tFar, tMax);

        if (tMin > tMax) {
            return false;
        }
    }

    return true;
}

template <typename T>
BoundingBoxRayIntersection2<T> BoundingBox<T, 2>::closestIntersection(
    const Ray2<T>& ray) const {
    BoundingBoxRayIntersection2<T> intersection;

    T tMin = 0;
    T tMax = std::numeric_limits<T>::max();

    const Vector2<T>& rayInvDir = ray.direction.rdiv(1);

    for (int i = 0; i < 2; ++i) {
        T tNear = (lowerCorner[i] - ray.origin[i]) * rayInvDir[i];
        T tFar = (upperCorner[i] - ray.origin[i]) * rayInvDir[i];

        if (tNear > tFar) {
            std::swap(tNear, tFar);
        }

        tMin = std::max(tNear, tMin);
        tMax = std::min(tFar, tMax);

        if (tMin > tMax) {
            intersection.isIntersecting = false;
            return intersection;
        }
    }

    intersection.isIntersecting = true;

    if (contains(ray.origin)) {
        intersection.tNear = tMax;
        intersection.tFar = std::numeric_limits<T>::max();
    } else {
        intersection.tNear = tMin;
        intersection.tFar = tMax;
    }

    return intersection;
}

template <typename T>
Vector2<T> BoundingBox<T, 2>::midPoint() const {
    return (upperCorner + lowerCorner) / static_cast<T>(2);
}

template <typename T>
T BoundingBox<T, 2>::diagonalLength() const {
    return (upperCorner - lowerCorner).length();
}

template <typename T>
T BoundingBox<T, 2>::diagonalLengthSquared() const {
    return (upperCorner - lowerCorner).lengthSquared();
}

template <typename T>
void BoundingBox<T, 2>::reset() {
    lowerCorner.x = std::numeric_limits<T>::max();
    lowerCorner.y = std::numeric_limits<T>::max();
    upperCorner.x = -std::numeric_limits<T>::max();
    upperCorner.y = -std::numeric_limits<T>::max();
}

template <typename T>
void BoundingBox<T, 2>::merge(const Vector2<T>& point) {
    lowerCorner.x = std::min(lowerCorner.x, point.x);
    lowerCorner.y = std::min(lowerCorner.y, point.y);
    upperCorner.x = std::max(upperCorner.x, point.x);
    upperCorner.y = std::max(upperCorner.y, point.y);
}

template <typename T>
void BoundingBox<T, 2>::merge(const BoundingBox& other) {
    lowerCorner.x = std::min(lowerCorner.x, other.lowerCorner.x);
    lowerCorner.y = std::min(lowerCorner.y, other.lowerCorner.y);
    upperCorner.x = std::max(upperCorner.x, other.upperCorner.x);
    upperCorner.y = std::max(upperCorner.y, other.upperCorner.y);
}

template <typename T>
void BoundingBox<T, 2>::expand(T delta) {
    lowerCorner -= delta;
    upperCorner += delta;
}

template <typename T>
Vector2<T> BoundingBox<T, 2>::corner(size_t idx) const {
    static const T h = static_cast<T>(1) / 2;
    static const Vector2<T> offset[4] = {
        {-h, -h}, {+h, -h}, {-h, +h}, {+h, +h}};

    return Vector2<T>(width(), height()) * offset[idx] + midPoint();
}

template <typename T>
Vector2<T> BoundingBox<T, 2>::clamp(const Vector2<T>& pt) const {
    return ::jet::clamp(pt, lowerCorner, upperCorner);
}

template <typename T>
bool BoundingBox<T, 2>::isEmpty() const {
    return (lowerCorner.x >= upperCorner.x || lowerCorner.y >= upperCorner.y);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_BOUNDING_BOX2_INL_H_
