// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/view_controller.h>

using namespace jet;
using namespace viz;

ViewController::ViewController(const CameraPtr& camera) : _camera(camera) {
    assert(_camera != nullptr);
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

void ViewController::resize(const Viewport& viewport) {
    _camera->resize(viewport);

    onResize(viewport);
}

const CameraPtr& ViewController::camera() const { return _camera; }

Event<ViewController*>& ViewController::onBasicCameraStateChanged() {
    return _basicCameraStateChangedEvent;
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

void ViewController::onResize(const Viewport& viewport) {
    UNUSED_VARIABLE(viewport);
}

void ViewController::setBasicCameraState(const BasicCameraState& newState) {
    _camera->setBasicCameraState(newState);

    _basicCameraStateChangedEvent(this);
}
