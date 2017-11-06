// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/persp_camera.h>
#include <jet.viz/spherical_view_controller.h>

using namespace jet;
using namespace viz;

static const double kRotateSpeedMultiplier = 0.01;
static const double kZoomSpeedMultiplier = 0.01;
static const double kPanSpeedMultiplier = 0.001;

SphericalViewController::SphericalViewController(const CameraPtr& camera)
    : ViewController(camera) {
    updateCamera();
}

SphericalViewController::~SphericalViewController() {}

void SphericalViewController::onKeyDown(const KeyEvent& keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void SphericalViewController::onKeyUp(const KeyEvent& keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void SphericalViewController::onPointerPressed(
    const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void SphericalViewController::onPointerHover(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void SphericalViewController::onPointerDragged(
    const PointerEvent& pointerEvent) {
    double deltaX = static_cast<double>(pointerEvent.deltaX());
    double deltaY = static_cast<double>(pointerEvent.deltaY());

    if (pointerEvent.modifierKey() == ModifierKey::Ctrl) {
        _azimuthalAngleInRadians -=
            kRotateSpeedMultiplier * _rotateSpeed * deltaX;
        _polarAngleInRadians -= kRotateSpeedMultiplier * _rotateSpeed * deltaY;
        _polarAngleInRadians = clamp(_polarAngleInRadians, 0.0, pi<double>());
    } else {
        BasicCameraState state = camera()->basicCameraState();
        Vector3D right = state.lookAt.cross(state.lookUp);

        // This should use unproject
        _origin += kPanSpeedMultiplier * _panSpeed *
                   (-deltaX * right + deltaY * state.lookUp);
    }

    updateCamera();
}

void SphericalViewController::onPointerReleased(
    const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void SphericalViewController::onMouseWheel(const PointerEvent& pointerEvent) {
    _radialDistance -=
        kZoomSpeedMultiplier * _zoomSpeed * pointerEvent.wheelData().deltaY;
    _radialDistance = std::max(_radialDistance, 0.0);

    updateCamera();
}

void SphericalViewController::onResize(const Viewport& viewport) {
    UNUSED_VARIABLE(viewport);
    updateCamera();
}

void SphericalViewController::updateCamera() {
    double x = _radialDistance * std::sin(_polarAngleInRadians) *
               std::sin(_azimuthalAngleInRadians);
    double y = _radialDistance * std::cos(_polarAngleInRadians);
    double z = _radialDistance * std::sin(_polarAngleInRadians) *
               std::cos(_azimuthalAngleInRadians);

    Vector3D positionInLocal = x * _basisX + y * _rotationAxis + z * _basisZ;

    BasicCameraState state = camera()->basicCameraState();

    state.origin = positionInLocal + _origin;
    state.lookAt = -positionInLocal.normalized();

    double upPolarAngleInRadians = pi<double>() / 2.0 - _polarAngleInRadians;
    double upAzimuthalAngleInRadians = pi<double>() + _azimuthalAngleInRadians;

    double upX =
        std::sin(upPolarAngleInRadians) * std::sin(upAzimuthalAngleInRadians);
    double upY = std::cos(upPolarAngleInRadians);
    double upZ =
        std::sin(upPolarAngleInRadians) * std::cos(upAzimuthalAngleInRadians);

    state.lookUp =
        (upX * _basisX + upY * _rotationAxis + upZ * _basisZ).normalized();

    setBasicCameraState(state);
}
