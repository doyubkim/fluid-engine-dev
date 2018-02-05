// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_BOUNDING_BOX3_INL_H_
#define INCLUDE_JET_DETAIL_BOUNDING_BOX3_INL_H_

#include <algorithm>
#include <limits>
#include <utility>  // just make cpplint happy..

namespace jet {

template <typename T>
BoundingBox<T, 3>::BoundingBox() {
    reset();
}

template <typename T>
BoundingBox<T, 3>::BoundingBox(const Vector3<T>& point1,
                               const Vector3<T>& point2) {
    lowerCorner.x = std::min(point1.x, point2.x);
    lowerCorner.y = std::min(point1.y, point2.y);
    lowerCorner.z = std::min(point1.z, point2.z);
    upperCorner.x = std::max(point1.x, point2.x);
    upperCorner.y = std::max(point1.y, point2.y);
    upperCorner.z = std::max(point1.z, point2.z);
}

template <typename T>
BoundingBox<T, 3>::BoundingBox(const BoundingBox& other)
    : lowerCorner(other.lowerCorner), upperCorner(other.upperCorner) {}

template <typename T>
T BoundingBox<T, 3>::width() const {
    return upperCorner.x - lowerCorner.x;
}

template <typename T>
T BoundingBox<T, 3>::height() const {
    return upperCorner.y - lowerCorner.y;
}

template <typename T>
T BoundingBox<T, 3>::depth() const {
    return upperCorner.z - lowerCorner.z;
}

template <typename T>
T BoundingBox<T, 3>::length(size_t axis) {
    return upperCorner[axis] - lowerCorner[axis];
}

template <typename T>
bool BoundingBox<T, 3>::overlaps(const BoundingBox& other) const {
    if (upperCorner.x < other.lowerCorner.x ||
        lowerCorner.x > other.upperCorner.x) {
        return false;
    }

    if (upperCorner.y < other.lowerCorner.y ||
        lowerCorner.y > other.upperCorner.y) {
        return false;
    }

    if (upperCorner.z < other.lowerCorner.z ||
        lowerCorner.z > other.upperCorner.z) {
        return false;
    }

    return true;
}

template <typename T>
bool BoundingBox<T, 3>::contains(const Vector3<T>& point) const {
    if (upperCorner.x < point.x || lowerCorner.x > point.x) {
        return false;
    }

    if (upperCorner.y < point.y || lowerCorner.y > point.y) {
        return false;
    }

    if (upperCorner.z < point.z || lowerCorner.z > point.z) {
        return false;
    }

    return true;
}

template <typename T>
bool BoundingBox<T, 3>::intersects(const Ray3<T>& ray) const {
    T tMin = 0;
    T tMax = std::numeric_limits<T>::max();
    const Vector3<T>& rayInvDir = ray.direction.rdiv(1);

    for (int i = 0; i < 3; ++i) {
        T tNear = (lowerCorner[i] - ray.origin[i]) * rayInvDir[i];
        T tFar = (upperCorner[i] - ray.origin[i]) * rayInvDir[i];

        if (tNear > tFar) std::swap(tNear, tFar);
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        if (tMin > tMax) return false;
    }

    return true;
}

template <typename T>
BoundingBoxRayIntersection3<T> BoundingBox<T, 3>::closestIntersection(
    const Ray3<T>& ray) const {
    BoundingBoxRayIntersection3<T> intersection;

    T tMin = 0;
    T tMax = std::numeric_limits<T>::max();
    const Vector3<T>& rayInvDir = ray.direction.rdiv(1);

    for (int i = 0; i < 3; ++i) {
        T tNear = (lowerCorner[i] - ray.origin[i]) * rayInvDir[i];
        T tFar = (upperCorner[i] - ray.origin[i]) * rayInvDir[i];

        if (tNear > tFar) std::swap(tNear, tFar);
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

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
Vector3<T> BoundingBox<T, 3>::midPoint() const {
    return (upperCorner + lowerCorner) / static_cast<T>(2);
}

template <typename T>
T BoundingBox<T, 3>::diagonalLength() const {
    return (upperCorner - lowerCorner).length();
}

template <typename T>
T BoundingBox<T, 3>::diagonalLengthSquared() const {
    return (upperCorner - lowerCorner).lengthSquared();
}

template <typename T>
void BoundingBox<T, 3>::reset() {
    lowerCorner.x = std::numeric_limits<T>::max();
    lowerCorner.y = std::numeric_limits<T>::max();
    lowerCorner.z = std::numeric_limits<T>::max();
    upperCorner.x = -std::numeric_limits<T>::max();
    upperCorner.y = -std::numeric_limits<T>::max();
    upperCorner.z = -std::numeric_limits<T>::max();
}

template <typename T>
void BoundingBox<T, 3>::merge(const Vector3<T>& point) {
    lowerCorner.x = std::min(lowerCorner.x, point.x);
    lowerCorner.y = std::min(lowerCorner.y, point.y);
    lowerCorner.z = std::min(lowerCorner.z, point.z);
    upperCorner.x = std::max(upperCorner.x, point.x);
    upperCorner.y = std::max(upperCorner.y, point.y);
    upperCorner.z = std::max(upperCorner.z, point.z);
}

template <typename T>
void BoundingBox<T, 3>::merge(const BoundingBox& other) {
    lowerCorner.x = std::min(lowerCorner.x, other.lowerCorner.x);
    lowerCorner.y = std::min(lowerCorner.y, other.lowerCorner.y);
    lowerCorner.z = std::min(lowerCorner.z, other.lowerCorner.z);
    upperCorner.x = std::max(upperCorner.x, other.upperCorner.x);
    upperCorner.y = std::max(upperCorner.y, other.upperCorner.y);
    upperCorner.z = std::max(upperCorner.z, other.upperCorner.z);
}

template <typename T>
void BoundingBox<T, 3>::expand(T delta) {
    lowerCorner -= delta;
    upperCorner += delta;
}

template <typename T>
Vector3<T> BoundingBox<T, 3>::corner(size_t idx) const {
    static const T h = static_cast<T>(1) / 2;
    static const Vector3<T> offset[8] = {
        {-h, -h, -h}, {+h, -h, -h}, {-h, +h, -h}, {+h, +h, -h},
        {-h, -h, +h}, {+h, -h, +h}, {-h, +h, +h}, {+h, +h, +h}};

    return Vector3<T>(width(), height(), depth()) * offset[idx] + midPoint();
}

template <typename T>
Vector3<T> BoundingBox<T, 3>::clamp(const Vector3<T>& pt) const {
    return ::jet::clamp(pt, lowerCorner, upperCorner);
}

template <typename T>
bool BoundingBox<T, 3>::isEmpty() const {
    return (lowerCorner.x >= upperCorner.x || lowerCorner.y >= upperCorner.y ||
            lowerCorner.z >= upperCorner.z);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_BOUNDING_BOX3_INL_H_
