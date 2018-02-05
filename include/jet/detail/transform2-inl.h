// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_TRANSFORM2_INL_H_
#define INCLUDE_JET_DETAIL_TRANSFORM2_INL_H_

#include <jet/transform2.h>

#include <algorithm>
#include <cmath>

namespace jet {

inline Transform2::Transform2() {
}

inline Transform2::Transform2(
    const Vector2D& translation,
    double orientation)
: _translation(translation)
, _orientation(orientation) {
    _cosAngle = std::cos(orientation);
    _sinAngle = std::sin(orientation);
}

inline const Vector2D& Transform2::translation() const {
    return _translation;
}

inline void Transform2::setTranslation(const Vector2D& translation) {
    _translation = translation;
}

inline double Transform2::orientation() const {
    return _orientation;
}

inline void Transform2::setOrientation(double orientation) {
    _orientation = orientation;
    _cosAngle = std::cos(orientation);
    _sinAngle = std::sin(orientation);
}

inline Vector2D Transform2::toLocal(const Vector2D& pointInWorld) const {
    // Convert to the local frame
    Vector2D xmt = pointInWorld - _translation;
    return Vector2D(
        _cosAngle * xmt.x + _sinAngle * xmt.y,
       -_sinAngle * xmt.x + _cosAngle * xmt.y);
}

inline Vector2D Transform2::toLocalDirection(const Vector2D& dirInWorld) const {
    // Convert to the local frame
    return Vector2D(
        _cosAngle * dirInWorld.x + _sinAngle * dirInWorld.y,
       -_sinAngle * dirInWorld.x + _cosAngle * dirInWorld.y);
}

inline Ray2D Transform2::toLocal(const Ray2D& rayInWorld) const {
    return Ray2D(
        toLocal(rayInWorld.origin),
        toLocalDirection(rayInWorld.direction));
}

inline BoundingBox2D Transform2::toLocal(
    const BoundingBox2D& bboxInWorld) const {
    BoundingBox2D bboxInLocal;
    for (int i = 0; i < 4; ++i) {
        auto cornerInLocal = toLocal(bboxInWorld.corner(i));
        bboxInLocal.lowerCorner
            = min(bboxInLocal.lowerCorner, cornerInLocal);
        bboxInLocal.upperCorner
            = max(bboxInLocal.upperCorner, cornerInLocal);
    }
    return bboxInLocal;
}

inline Vector2D Transform2::toWorld(const Vector2D& pointInLocal) const {
    // Convert to the world frame
    return Vector2D(
        _cosAngle * pointInLocal.x - _sinAngle * pointInLocal.y
            + _translation.x,
        _sinAngle * pointInLocal.x + _cosAngle * pointInLocal.y
            + _translation.y);
}

inline Vector2D Transform2::toWorldDirection(const Vector2D& dirInLocal) const {
    // Convert to the world frame
    return Vector2D(
        _cosAngle * dirInLocal.x - _sinAngle * dirInLocal.y,
        _sinAngle * dirInLocal.x + _cosAngle * dirInLocal.y);
}

inline Ray2D Transform2::toWorld(const Ray2D& rayInLocal) const {
    return Ray2D(
        toWorld(rayInLocal.origin),
        toWorldDirection(rayInLocal.direction));
}

inline BoundingBox2D Transform2::toWorld(
    const BoundingBox2D& bboxInLocal) const {
    BoundingBox2D bboxInWorld;
    for (int i = 0; i < 4; ++i) {
        auto cornerInWorld = toWorld(bboxInLocal.corner(i));
        bboxInWorld.lowerCorner
            = min(bboxInWorld.lowerCorner, cornerInWorld);
        bboxInWorld.upperCorner
            = max(bboxInWorld.upperCorner, cornerInWorld);
    }
    return bboxInWorld;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_TRANSFORM2_INL_H_
