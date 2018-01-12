// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/persp_camera.h>
#include <jet.viz/pitch_yaw_view_controller.h>

using namespace jet;
using namespace viz;

constexpr double kRotateSpeedMultiplier = 0.01;
constexpr double kZoomSpeedMultiplier = 0.01;
constexpr double kPanSpeedMultiplier = 0.001;
constexpr double kMinRadialDistance = 1e-3;

PitchYawViewController::PitchYawViewController(const CameraPtr& camera,
                                               const Vector3D& rotationOrigin)
    : ViewController(camera), _origin(rotationOrigin) {
    _radialDistance =
        std::max((camera->basicCameraState().origin - _origin).length(),
                 kMinRadialDistance);
    updateCamera();
}

PitchYawViewController::~PitchYawViewController() {}

void PitchYawViewController::onKeyDown(const KeyEvent& keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void PitchYawViewController::onKeyUp(const KeyEvent& keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void PitchYawViewController::onPointerPressed(
    const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void PitchYawViewController::onPointerHover(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void PitchYawViewController::onPointerDragged(
    const PointerEvent& pointerEvent) {
    double deltaX = static_cast<double>(pointerEvent.deltaX());
    double deltaY = static_cast<double>(pointerEvent.deltaY());

    if (pointerEvent.modifierKey() == ModifierKey::kCtrl) {
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

void PitchYawViewController::onPointerReleased(
    const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void PitchYawViewController::onMouseWheel(const PointerEvent& pointerEvent) {
    _radialDistance -=
        kZoomSpeedMultiplier * _zoomSpeed * pointerEvent.wheelData().deltaY;
    _radialDistance = std::max(_radialDistance, kMinRadialDistance);

    updateCamera();
}

void PitchYawViewController::onResize(const Viewport& viewport) {
    UNUSED_VARIABLE(viewport);
    updateCamera();
}

void PitchYawViewController::updateCamera() {
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
