// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_BOUNDING_BOX_INL_H_
#define INCLUDE_JET_DETAIL_BOUNDING_BOX_INL_H_

#include <algorithm>
#include <limits>

namespace jet {

template <typename T, size_t N>
BoundingBox<T, N>::BoundingBox() {
    reset();
}

template <typename T, size_t N>
BoundingBox<T, N>::BoundingBox(const VectorType& point1,
                               const VectorType& point2) {
    lowerCorner = min(point1, point2);
    upperCorner = max(point1, point2);
}

template <typename T, size_t N>
BoundingBox<T, N>::BoundingBox(const BoundingBox& other)
    : lowerCorner(other.lowerCorner), upperCorner(other.upperCorner) {}

template <typename T, size_t N>
T BoundingBox<T, N>::width() const {
    return upperCorner[0] - lowerCorner[0];
}

template <typename T, size_t N>
template <typename U>
std::enable_if_t<(N > 1), U> BoundingBox<T, N>::height() const {
    return upperCorner[1] - lowerCorner[1];
}

template <typename T, size_t N>
template <typename U>
std::enable_if_t<(N > 2), U> BoundingBox<T, N>::depth() const {
    return upperCorner[2] - lowerCorner[2];
}

template <typename T, size_t N>
T BoundingBox<T, N>::length(size_t axis) {
    return upperCorner[axis] - lowerCorner[axis];
}

template <typename T, size_t N>
bool BoundingBox<T, N>::overlaps(const BoundingBox& other) const {
    for (size_t i = 0; i < N; ++i) {
        if (upperCorner[i] < other.lowerCorner[i] ||
            lowerCorner[i] > other.upperCorner[i]) {
            return false;
        }
    }

    return true;
}

template <typename T, size_t N>
bool BoundingBox<T, N>::contains(const VectorType& point) const {
    for (size_t i = 0; i < N; ++i) {
        if (upperCorner[i] < point[i] || lowerCorner[i] > point[i]) {
            return false;
        }
    }

    return true;
}

template <typename T, size_t N>
bool BoundingBox<T, N>::intersects(const RayType& ray) const {
    T tMin = 0;
    T tMax = std::numeric_limits<T>::max();
    const VectorType& rayInvDir = T(1) / ray.direction;

    for (size_t i = 0; i < N; ++i) {
        T tNear = (lowerCorner[i] - ray.origin[i]) * rayInvDir[i];
        T tFar = (upperCorner[i] - ray.origin[i]) * rayInvDir[i];

        if (tNear > tFar) std::swap(tNear, tFar);
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        if (tMin > tMax) return false;
    }

    return true;
}

template <typename T, size_t N>
BoundingBoxRayIntersection<T> BoundingBox<T, N>::closestIntersection(
    const RayType& ray) const {
    BoundingBoxRayIntersection<T> intersection;

    T tMin = 0;
    T tMax = std::numeric_limits<T>::max();
    const VectorType& rayInvDir = T(1) / ray.direction;

    for (int i = 0; i < N; ++i) {
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

template <typename T, size_t N>
typename BoundingBox<T, N>::VectorType BoundingBox<T, N>::midPoint() const {
    return (upperCorner + lowerCorner) / static_cast<T>(2);
}

template <typename T, size_t N>
T BoundingBox<T, N>::diagonalLength() const {
    return VectorType(upperCorner - lowerCorner).length();
}

template <typename T, size_t N>
T BoundingBox<T, N>::diagonalLengthSquared() const {
    return VectorType(upperCorner - lowerCorner).lengthSquared();
}

template <typename T, size_t N>
void BoundingBox<T, N>::reset() {
    lowerCorner = VectorType::makeConstant(std::numeric_limits<T>::max());
    upperCorner = VectorType::makeConstant(-std::numeric_limits<T>::max());
}

template <typename T, size_t N>
void BoundingBox<T, N>::merge(const VectorType& point) {
    lowerCorner = min(lowerCorner, point);
    upperCorner = max(upperCorner, point);
}

template <typename T, size_t N>
void BoundingBox<T, N>::merge(const BoundingBox& other) {
    lowerCorner = min(lowerCorner, other.lowerCorner);
    upperCorner = max(upperCorner, other.upperCorner);
}

template <typename T, size_t N>
void BoundingBox<T, N>::expand(T delta) {
    lowerCorner -= delta;
    upperCorner += delta;
}

template <typename T, size_t N>
typename BoundingBox<T, N>::VectorType BoundingBox<T, N>::corner(
    size_t idx) const {
    VectorType ret;
    for (size_t i = 0; i < N; ++i) {
        ret[i] = lowerCorner[i] + (((kOneSize << i) & idx) != 0) *
                                      (upperCorner[i] - lowerCorner[i]);
    }
    return ret;
}

template <typename T, size_t N>
typename BoundingBox<T, N>::VectorType BoundingBox<T, N>::clamp(
    const VectorType& pt) const {
    return ::jet::clamp(pt, lowerCorner, upperCorner);
}

template <typename T, size_t N>
bool BoundingBox<T, N>::isEmpty() const {
    for (size_t i = 0; i < N; ++i) {
        if (lowerCorner[i] >= upperCorner[i]) {
            return true;
        }
    }
    return false;
}

template <typename T, size_t N>
template <typename U>
BoundingBox<U, N> BoundingBox<T, N>::castTo() const {
    return BoundingBox<U, N>{lowerCorner.template castTo<U>(),
                             upperCorner.template castTo<U>()};
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_BOUNDING_BOX_INL_H_
