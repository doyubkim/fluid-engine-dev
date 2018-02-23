// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/camera.h>

using namespace jet;
using namespace viz;

Camera::Camera()
    : Camera(Vector3D(0, 0, 1), Vector3D(0, 0, -1), Vector3D(0, 1, 0), 0.1,
             10.0, Viewport()) {
    updateMatrix();
}

Camera::Camera(const Vector3D& origin, const Vector3D& lookAt,
               const Vector3D& lookUp, double nearClipPlane,
               double farClipPlane, const Viewport& viewport) {
    _matrix = Matrix4x4D::makeIdentity();

    _state.origin = origin;
    _state.lookAt = lookAt;
    _state.lookUp = lookUp;
    _state.nearClipPlane = nearClipPlane;
    _state.farClipPlane = farClipPlane;
    _state.viewport = viewport;

    _state.lookAt.normalize();
    _state.lookUp.normalize();

    updateMatrix();
}

Camera::~Camera() {}

void Camera::resize(const Viewport& viewport) {
    _state.viewport = viewport;

    onResize(viewport);

    updateMatrix();
}

void Camera::onResize(const Viewport& viewport) { UNUSED_VARIABLE(viewport); }

const Matrix4x4D& Camera::matrix() const { return _matrix; }

const BasicCameraState& Camera::basicCameraState() const { return _state; }

void Camera::setBasicCameraState(const BasicCameraState& state) {
    _state = state;

    updateMatrix();
}

void Camera::updateMatrix() {
    // Do nothing
}
