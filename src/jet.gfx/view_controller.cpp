// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/view_controller.h>

using namespace jet;
using namespace gfx;

ViewController::ViewController(const CameraPtr& camera) : _camera(camera) {
    JET_ASSERT(_camera != nullptr);
}

ViewController::~ViewController() {}

void ViewController::keyDown(const KeyEvent& keyEvent) { onKeyDown(keyEvent); }

void ViewController::keyUp(const KeyEvent& keyEvent) { onKeyUp(keyEvent); }

void ViewController::pointerPressed(const PointerEvent& pointerEvent) {
    onPointerPressed(pointerEvent);
}

void ViewController::pointerHover(const PointerEvent& pointerEvent) {
    onPointerHover(pointerEvent);
}

void ViewController::pointerDragged(const PointerEvent& pointerEvent) {
    onPointerDragged(pointerEvent);
}

void ViewController::pointerReleased(const PointerEvent& pointerEvent) {
    onPointerReleased(pointerEvent);
}

void ViewController::mouseWheel(const PointerEvent& pointerEvent) {
    onMouseWheel(pointerEvent);
}

void ViewController::setViewport(const Viewport& viewport) {
    CameraState newState = _camera->state;
    newState.viewport = viewport;
    setCameraState(newState);
}

const CameraPtr& ViewController::camera() const { return _camera; }

Event<ViewController*>& ViewController::onCameraStateChanged() {
    return _cameraStateChangedEvent;
}

void ViewController::onKeyDown(const KeyEvent& keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void ViewController::onKeyUp(const KeyEvent& keyEvent) {
    UNUSED_VARIABLE(keyEvent);
}

void ViewController::onPointerPressed(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void ViewController::onPointerHover(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void ViewController::onPointerDragged(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void ViewController::onPointerReleased(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void ViewController::onMouseWheel(const PointerEvent& pointerEvent) {
    UNUSED_VARIABLE(pointerEvent);
}

void ViewController::setCameraState(const CameraState& newState) {
    if (_camera->state != newState) {
        _camera->state = newState;

        JET_DEBUG << "Camera state changed - "
                  << "viewport width: " << newState.viewport.width << ", "
                  << "viewport height: " << newState.viewport.height;

        _cameraStateChangedEvent(this);
    }
}
