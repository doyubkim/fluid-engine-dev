// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/ortho_view_controller.h>
#include <jet/matrix.h>

namespace jet {
namespace gfx {

static const float kZoomSpeedMultiplier = 0.1f;

OrthoViewController::OrthoViewController(const OrthoCameraPtr &camera)
    : ViewController(camera) {
    _origin = camera->state.origin;
    _basisY = camera->state.lookUp;
    _basisX = camera->state.lookAt.cross(_basisY);
    _viewHeight = camera->height();

    updateCamera();
}

OrthoViewController::~OrthoViewController() {}

void OrthoViewController::onKeyDown(const KeyEvent &keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void OrthoViewController::onKeyUp(const KeyEvent &keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void OrthoViewController::onPointerPressed(const PointerEvent &pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void OrthoViewController::onPointerHover(const PointerEvent &pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void OrthoViewController::onPointerDragged(const PointerEvent &pointerEvent) {
    float deltaX = pointerEvent.deltaX();
    float deltaY = pointerEvent.deltaY();

    if (enableRotation && pointerEvent.modifierKey() == ModifierKey::Ctrl) {
        Vector2F center = camera()->state.viewport.center();
        Vector2F offset(pointerEvent.x() - center.x,
                        center.y - pointerEvent.y());
        float startAngle = std::atan2(offset.y, offset.x);
        float endAngle = std::atan2(offset.y - deltaY, offset.x + deltaX);

        _viewRotateAngleInRadians += endAngle - startAngle;
    } else if (enablePan) {
        OrthoCameraPtr orthoCamera =
            std::dynamic_pointer_cast<OrthoCamera>(camera());

        const float scaleX =
            orthoCamera->width() / camera()->state.viewport.width;
        const float scaleY =
            orthoCamera->height() / camera()->state.viewport.height;

        const auto right = std::cos(_viewRotateAngleInRadians) * _basisX +
                           -std::sin(_viewRotateAngleInRadians) * _basisY;
        const auto up = std::sin(_viewRotateAngleInRadians) * _basisX +
                        std::cos(_viewRotateAngleInRadians) * _basisY;

        _origin += -(_panSpeed * scaleX * deltaX * right) +
                   (_panSpeed * scaleY * deltaY * up);
    }

    updateCamera();
}

void OrthoViewController::onPointerReleased(const PointerEvent &pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void OrthoViewController::onMouseWheel(const PointerEvent &pointerEvent) {
    if (enableZoom) {
        OrthoCameraPtr orthoCamera =
            std::dynamic_pointer_cast<OrthoCamera>(camera());

        const float scale = pow(0.5f, kZoomSpeedMultiplier * _zoomSpeed *
                                          pointerEvent.wheelData().deltaY);
        const auto screen = camera()->state.viewport;
        const float aspectRatio = screen.aspectRatio();
        const float oldViewHeight = _viewHeight;
        const float oldViewWidth = oldViewHeight * aspectRatio;
        const float newViewHeight = _viewHeight * scale;
        const float newViewWidth = newViewHeight * aspectRatio;

        const float sx = pointerEvent.x();
        const float sy = screen.height - pointerEvent.y();

        const float newCameraLeft =
            sx / screen.width * (1.0f - scale) * oldViewWidth +
            orthoCamera->left;
        const float newCameraBottom =
            sy / screen.height * (1.0f - scale) * oldViewHeight +
            orthoCamera->bottom;

        const float newCameraRight = newCameraLeft + newViewWidth;
        const float newCameraTop = newCameraBottom + newViewHeight;

        _viewHeight = newViewHeight;
        orthoCamera->left = newCameraLeft;
        orthoCamera->right = newCameraRight;
        orthoCamera->bottom = newCameraBottom;
        orthoCamera->top = newCameraTop;

        updateCamera();
    }
}

void OrthoViewController::updateCamera() {
    OrthoCameraPtr orthoCamera =
        std::dynamic_pointer_cast<OrthoCamera>(camera());
    CameraState state = orthoCamera->state;

    state.origin = _origin;
    state.lookUp = std::sin(_viewRotateAngleInRadians) * _basisX +
                   std::cos(_viewRotateAngleInRadians) * _basisY;

    float oldHeight = orthoCamera->height();
    float scale = _viewHeight / oldHeight;
    float newHalfHeight = 0.5f * scale * oldHeight;
    float newHalfWidth = (preserveAspectRatio)
                             ? newHalfHeight * state.viewport.aspectRatio()
                             : 0.5f * scale * orthoCamera->width();
    Vector2F center = orthoCamera->center();
    float newLeft = center.x - newHalfWidth;
    float newRight = center.x + newHalfWidth;
    float newBottom = center.y - newHalfHeight;
    float newTop = center.y + newHalfHeight;

    orthoCamera->left = newLeft;
    orthoCamera->right = newRight;
    orthoCamera->bottom = newBottom;
    orthoCamera->top = newTop;

    setCameraState(state);
}

}  // namespace gfx
}  // namespace jet
