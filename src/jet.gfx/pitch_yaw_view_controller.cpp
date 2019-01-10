// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/persp_camera.h>
#include <jet.gfx/pitch_yaw_view_controller.h>

namespace jet {
namespace gfx {

constexpr float kRotateSpeedMultiplier = 0.01f;
constexpr float kZoomSpeedMultiplier = 0.01f;
constexpr float kPanSpeedMultiplier = 0.001f;
constexpr float kMinRadialDistance = 1e-3f;

PitchYawViewController::PitchYawViewController(const CameraPtr &camera,
                                               const Vector3F &rotationOrigin)
    : ViewController(camera), _origin(rotationOrigin) {
    _radialDistance =
        std::max((camera->state.origin - _origin).length(), kMinRadialDistance);
    updateCamera();
}

PitchYawViewController::~PitchYawViewController() {}

void PitchYawViewController::onKeyDown(const KeyEvent &keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void PitchYawViewController::onKeyUp(const KeyEvent &keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void PitchYawViewController::onPointerPressed(
    const PointerEvent &pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void PitchYawViewController::onPointerHover(const PointerEvent &pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void PitchYawViewController::onPointerDragged(
    const PointerEvent &pointerEvent) {
    float deltaX = pointerEvent.deltaX();
    float deltaY = pointerEvent.deltaY();

    if (pointerEvent.modifierKey() == ModifierKey::Ctrl) {
        _azimuthalAngleInRadians -=
            kRotateSpeedMultiplier * _rotateSpeed * deltaX;
        _polarAngleInRadians -= kRotateSpeedMultiplier * _rotateSpeed * deltaY;
        _polarAngleInRadians = clamp(_polarAngleInRadians, 0.0f, kPiF);
    } else {
        CameraState state = camera()->state;
        Vector3F right = state.lookAt.cross(state.lookUp);

        // TODO: This should use unproject
        _origin += kPanSpeedMultiplier * _panSpeed *
                   (-deltaX * right + deltaY * state.lookUp);
    }

    updateCamera();
}

void PitchYawViewController::onPointerReleased(
    const PointerEvent &pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void PitchYawViewController::onMouseWheel(const PointerEvent &pointerEvent) {
    _radialDistance -=
        kZoomSpeedMultiplier * _zoomSpeed * pointerEvent.wheelData().deltaY;
    _radialDistance = std::max(_radialDistance, kMinRadialDistance);

    updateCamera();
}

void PitchYawViewController::updateCamera() {
    float x = _radialDistance * std::sin(_polarAngleInRadians) *
              std::sin(_azimuthalAngleInRadians);
    float y = _radialDistance * std::cos(_polarAngleInRadians);
    float z = _radialDistance * std::sin(_polarAngleInRadians) *
              std::cos(_azimuthalAngleInRadians);

    Vector3F positionInLocal = x * _basisX + y * _rotationAxis + z * _basisZ;

    CameraState state = camera()->state;

    state.origin = positionInLocal + _origin;
    state.lookAt = -positionInLocal.normalized();

    float upPolarAngleInRadians = kHalfPiF - _polarAngleInRadians;
    float upAzimuthalAngleInRadians = kPiF + _azimuthalAngleInRadians;

    float upX =
        std::sin(upPolarAngleInRadians) * std::sin(upAzimuthalAngleInRadians);
    float upY = std::cos(upPolarAngleInRadians);
    float upZ =
        std::sin(upPolarAngleInRadians) * std::cos(upAzimuthalAngleInRadians);

    state.lookUp =
        (upX * _basisX + upY * _rotationAxis + upZ * _basisZ).normalized();

    setCameraState(state);
}

}  // namespace gfx
}  // namespace jet
