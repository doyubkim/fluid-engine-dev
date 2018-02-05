// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_TRANSFORM3_INL_H_
#define INCLUDE_JET_DETAIL_TRANSFORM3_INL_H_

#include <jet/transform3.h>

#include <algorithm>
#include <cmath>

namespace jet {

inline Transform3::Transform3() {
}

inline Transform3::Transform3(
    const Vector3D& translation,
    const QuaternionD& orientation) {
    setTranslation(translation);
    setOrientation(orientation);
}

inline const Vector3D& Transform3::translation() const {
    return _translation;
}

inline void Transform3::setTranslation(const Vector3D& translation) {
    _translation = translation;
}

inline const QuaternionD& Transform3::orientation() const {
    return _orientation;
}

inline void Transform3::setOrientation(const QuaternionD& orientation) {
    _orientation = orientation;
    _orientationMat3 = orientation.matrix3();
    _inverseOrientationMat3 = orientation.inverse().matrix3();
}

inline Vector3D Transform3::toLocal(const Vector3D& pointInWorld) const {
    return _inverseOrientationMat3 * (pointInWorld - _translation);
}

inline Vector3D Transform3::toLocalDirection(const Vector3D& dirInWorld) const {
    return _inverseOrientationMat3 * dirInWorld;
}

inline Ray3D Transform3::toLocal(const Ray3D& rayInWorld) const {
    return Ray3D(
        toLocal(rayInWorld.origin),
        toLocalDirection(rayInWorld.direction));
}

inline BoundingBox3D Transform3::toLocal(
    const BoundingBox3D& bboxInWorld) const {
    BoundingBox3D bboxInLocal;
    for (int i = 0; i < 8; ++i) {
        auto cornerInLocal = toLocal(bboxInWorld.corner(i));
        bboxInLocal.lowerCorner
            = min(bboxInLocal.lowerCorner, cornerInLocal);
        bboxInLocal.upperCorner
            = max(bboxInLocal.upperCorner, cornerInLocal);
    }
    return bboxInLocal;
}

inline Vector3D Transform3::toWorld(const Vector3D& pointInLocal) const {
    return (_orientationMat3 * pointInLocal) + _translation;
}

inline Vector3D Transform3::toWorldDirection(
    const Vector3D& dirInLocal) const {
    return _orientationMat3 * dirInLocal;
}

inline Ray3D Transform3::toWorld(const Ray3D& rayInLocal) const {
    return Ray3D(
        toWorld(rayInLocal.origin),
        toWorldDirection(rayInLocal.direction));
}

inline BoundingBox3D Transform3::toWorld(
    const BoundingBox3D& bboxInLocal) const {
    BoundingBox3D bboxInWorld;
    for (int i = 0; i < 8; ++i) {
        auto cornerInWorld = toWorld(bboxInLocal.corner(i));
        bboxInWorld.lowerCorner
            = min(bboxInWorld.lowerCorner, cornerInWorld);
        bboxInWorld.upperCorner
            = max(bboxInWorld.upperCorner, cornerInWorld);
    }
    return bboxInWorld;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_TRANSFORM3_INL_H_
