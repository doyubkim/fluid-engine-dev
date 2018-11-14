// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/transform.h>

namespace jet {

// MARK: Orientation2

Orientation<2>::Orientation() : Orientation(0.0) {}

Orientation<2>::Orientation(double angleInRadian) {
    setRotation(angleInRadian);
}

double Orientation<2>::rotation() const { return _angle; }

void Orientation<2>::setRotation(double angleInRadian) {
    _angle = angleInRadian;
    _cosAngle = std::cos(angleInRadian);
    _sinAngle = std::sin(angleInRadian);
}

Vector2D Orientation<2>::toLocal(const Vector2D& pointInWorld) const {
    // Convert to the local frame
    return Vector2D(_cosAngle * pointInWorld.x + _sinAngle * pointInWorld.y,
                    -_sinAngle * pointInWorld.x + _cosAngle * pointInWorld.y);
}

Vector2D Orientation<2>::toWorld(const Vector2D& pointInLocal) const {
    // Convert to the world frame
    return Vector2D(_cosAngle * pointInLocal.x - _sinAngle * pointInLocal.y,
                    _sinAngle * pointInLocal.x + _cosAngle * pointInLocal.y);
}

// MARK: Orientation3

Orientation<3>::Orientation() {}

Orientation<3>::Orientation(const QuaternionD& quat) { setRotation(quat); }

const QuaternionD& Orientation<3>::rotation() const { return _quat; }

void Orientation<3>::setRotation(const QuaternionD& quat) {
    _quat = quat;
    _rotationMat3 = quat.matrix3();
    _inverseRotationMat3 = quat.inverse().matrix3();
}

Vector3D Orientation<3>::toLocal(const Vector3D& pointInWorld) const {
    return _inverseRotationMat3 * pointInWorld;
}

Vector3D Orientation<3>::toWorld(const Vector3D& pointInLocal) const {
    return _rotationMat3 * pointInLocal;
}

// MARK: Transform2 and 3

template <size_t N>
Transform<N>::Transform() {}

template <size_t N>
Transform<N>::Transform(const Vector<double, N>& translation,
                        const Orientation<N>& orientation) {
    setTranslation(translation);
    setOrientation(orientation);
}

template <size_t N>
const Vector<double, N>& Transform<N>::translation() const {
    return _translation;
}

template <size_t N>
void Transform<N>::setTranslation(const Vector<double, N>& translation) {
    _translation = translation;
}

template <size_t N>
const Orientation<N>& Transform<N>::orientation() const {
    return _orientation;
}

template <size_t N>
void Transform<N>::setOrientation(const Orientation<N>& orientation) {
    _orientation = orientation;
}

template <size_t N>
Vector<double, N> Transform<N>::toLocal(
    const Vector<double, N>& pointInWorld) const {
    return _orientation.toLocal(pointInWorld - _translation);
}

template <size_t N>
Vector<double, N> Transform<N>::toLocalDirection(
    const Vector<double, N>& dirInWorld) const {
    return _orientation.toLocal(dirInWorld);
}

template <size_t N>
Ray<double, N> Transform<N>::toLocal(const Ray<double, N>& rayInWorld) const {
    return Ray<double, N>(toLocal(rayInWorld.origin),
                          toLocalDirection(rayInWorld.direction));
}

template <size_t N>
BoundingBox<double, N> Transform<N>::toLocal(
    const BoundingBox<double, N>& bboxInWorld) const {
    BoundingBox<double, N> bboxInLocal;
    int numCorners = 2 << N;
    for (int i = 0; i < numCorners; ++i) {
        auto cornerInLocal = toLocal(bboxInWorld.corner(i));
        bboxInLocal.lowerCorner = min(bboxInLocal.lowerCorner, cornerInLocal);
        bboxInLocal.upperCorner = max(bboxInLocal.upperCorner, cornerInLocal);
    }
    return bboxInLocal;
}

template <size_t N>
Vector<double, N> Transform<N>::toWorld(
    const Vector<double, N>& pointInLocal) const {
    return _orientation.toWorld(pointInLocal) + _translation;
}

template <size_t N>
Vector<double, N> Transform<N>::toWorldDirection(
    const Vector<double, N>& dirInLocal) const {
    return _orientation.toWorld(dirInLocal);
}

template <size_t N>
Ray<double, N> Transform<N>::toWorld(const Ray<double, N>& rayInLocal) const {
    return Ray<double, N>(toWorld(rayInLocal.origin),
                          toWorldDirection(rayInLocal.direction));
}

template <size_t N>
BoundingBox<double, N> Transform<N>::toWorld(
    const BoundingBox<double, N>& bboxInLocal) const {
    BoundingBox<double, N> bboxInWorld;
    int numCorners = 2 << N;
    for (int i = 0; i < numCorners; ++i) {
        auto cornerInWorld = toWorld(bboxInLocal.corner(i));
        bboxInWorld.lowerCorner = min(bboxInWorld.lowerCorner, cornerInWorld);
        bboxInWorld.upperCorner = max(bboxInWorld.upperCorner, cornerInWorld);
    }
    return bboxInWorld;
}

template class Transform<2>;
template class Transform<3>;

}  // namespace jet