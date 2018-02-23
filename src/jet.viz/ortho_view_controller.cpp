// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/ortho_view_controller.h>
#include <jet/matrix2x2.h>

using namespace jet;
using namespace viz;

static const double kZoomSpeedMultiplier = 0.1;

OrthoViewController::OrthoViewController(const OrthoCameraPtr& camera)
    : ViewController(camera) {
    _origin = camera->basicCameraState().origin;
    _basisY = camera->basicCameraState().lookUp;
    _basisX = camera->basicCameraState().lookAt.cross(_basisY);
    _viewHeight = camera->height();
}

OrthoViewController::~OrthoViewController() {}

void OrthoViewController::onKeyDown(const KeyEvent& keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void OrthoViewController::onKeyUp(const KeyEvent& keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void OrthoViewController::onPointerPressed(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void OrthoViewController::onPointerHover(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void OrthoViewController::onPointerDragged(const PointerEvent& pointerEvent) {
    double deltaX = pointerEvent.deltaX();
    double deltaY = pointerEvent.deltaY();

    if (enableRotation && pointerEvent.modifierKey() == ModifierKey::kCtrl) {
        Vector2D center = camera()->basicCameraState().viewport.center();
        Vector2D offset(pointerEvent.x() - center.x,
                        center.y - pointerEvent.y());
        double startAngle = std::atan2(offset.y, offset.x);
        double endAngle = std::atan2(offset.y - deltaY, offset.x + deltaX);

        _viewRotateAngleInRadians += endAngle - startAngle;
    } else if (enablePan) {
        OrthoCameraPtr orthoCamera =
            std::dynamic_pointer_cast<OrthoCamera>(camera());

        const double scaleX =
            orthoCamera->width() / camera()->basicCameraState().viewport.width;
        const double scaleY = orthoCamera->height() /
                              camera()->basicCameraState().viewport.height;

        const auto right = std::cos(_viewRotateAngleInRadians) * _basisX +
                           -std::sin(_viewRotateAngleInRadians) * _basisY;
        const auto up = std::sin(_viewRotateAngleInRadians) * _basisX +
                        std::cos(_viewRotateAngleInRadians) * _basisY;

        _origin += -(_panSpeed * scaleX * deltaX * right) +
                   (_panSpeed * scaleY * deltaY * up);
    }

    updateCamera();
}

void OrthoViewController::onPointerReleased(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void OrthoViewController::onMouseWheel(const PointerEvent& pointerEvent) {
    if (enableZoom) {
        OrthoCameraPtr orthoCamera =
            std::dynamic_pointer_cast<OrthoCamera>(camera());

        const double scale = pow(0.5, kZoomSpeedMultiplier * _zoomSpeed *
                                          pointerEvent.wheelData().deltaY);
        const auto screen = camera()->basicCameraState().viewport;
        const double aspectRatio = screen.aspectRatio();
        const double oldViewHeight = _viewHeight;
        const double oldViewWidth = oldViewHeight * aspectRatio;
        const double newViewHeight = _viewHeight * scale;
        const double newViewWidth = newViewHeight * aspectRatio;

        const double sx = pointerEvent.x();
        const double sy = screen.height - pointerEvent.y();

        const double newCameraLeft =
            sx / screen.width * (1.0 - scale) * oldViewWidth +
            orthoCamera->left();
        const double newCameraBottom =
            sy / screen.height * (1.0 - scale) * oldViewHeight +
            orthoCamera->bottom();

        const double newCameraRight = newCameraLeft + newViewWidth;
        const double newCameraTop = newCameraBottom + newViewHeight;

        _viewHeight = newViewHeight;
        orthoCamera->setLeft(newCameraLeft);
        orthoCamera->setRight(newCameraRight);
        orthoCamera->setBottom(newCameraBottom);
        orthoCamera->setTop(newCameraTop);

        updateCamera();
    }
}

void OrthoViewController::onResize(const Viewport& viewport) {
    UNUSED_VARIABLE(viewport);
    updateCamera();
}

void OrthoViewController::updateCamera() {
    OrthoCameraPtr orthoCamera =
        std::dynamic_pointer_cast<OrthoCamera>(camera());
    BasicCameraState state = orthoCamera->basicCameraState();

    state.origin = _origin;
    state.lookUp = std::sin(_viewRotateAngleInRadians) * _basisX +
                   std::cos(_viewRotateAngleInRadians) * _basisY;

    double oldHeight = orthoCamera->height();
    double scale = _viewHeight / oldHeight;
    double newHalfHeight = 0.5 * scale * oldHeight;
    double newHalfWidth = (preserveAspectRatio)
                              ? newHalfHeight * state.viewport.aspectRatio()
                              : 0.5 * scale * orthoCamera->width();
    Vector2D center = orthoCamera->center();
    double newLeft = center.x - newHalfWidth;
    double newRight = center.x + newHalfWidth;
    double newBottom = center.y - newHalfHeight;
    double newTop = center.y + newHalfHeight;

    orthoCamera->setLeft(newLeft);
    orthoCamera->setRight(newRight);
    orthoCamera->setBottom(newBottom);
    orthoCamera->setTop(newTop);

    setBasicCameraState(state);
}
