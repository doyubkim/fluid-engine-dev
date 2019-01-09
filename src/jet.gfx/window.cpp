// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/window.h>

namespace jet {
namespace gfx {

void Window::setSwapInterval(int interval) { UNUSED_VARIABLE(interval); }

Vector2F Window::displayScalingFactor() const {
    return elemDiv(framebufferSize().castTo<float>(),
                   windowSize().castTo<float>());
}

void Window::requestRender(unsigned int numFrames) {
    UNUSED_VARIABLE(numFrames);
}

const ViewControllerPtr &Window::viewController() const {
    return _viewController;
}

void Window::setViewController(const ViewControllerPtr &viewController) {
    // Detach event handler from old view controller
    if (_onCameraStateChangedEventToken != kEmptyEventToken &&
        _viewController != nullptr) {
        _viewController->onCameraStateChanged() -=
            _onCameraStateChangedEventToken;
    }

    _viewController = viewController;

    _onCameraStateChangedEventToken = viewController->onCameraStateChanged() +=
        [this](ViewController *) {
            // When basic camera state changes, update the view.
            requestRender(1);
            return true;
        };

    _renderer->setCamera(_viewController->camera());
}

const RendererPtr &Window::renderer() const { return _renderer; }

bool Window::isUpdateEnabled() const { return _isUpdateEnabled; }

void Window::setIsUpdateEnabled(bool enabled) {
    _isUpdateEnabled = enabled;
    onUpdateEnabled(enabled);
}

Event<Window *> &Window::onUpdateEvent() { return _onUpdateEvent; }

Event<Window *> &Window::onGuiEvent() { return _onGuiEvent; }

Event<Window *, const Vector2I &> &Window::onWindowResizedEvent() {
    return _onWindowResizedEvent;
}

Event<Window *, const Vector2I &> &Window::onWindowMovedEvent() {
    return _onWindowMovedEvent;
}

Event<Window *, const KeyEvent &> &Window::onKeyDownEvent() {
    return _onKeyDownEvent;
}

Event<Window *, const KeyEvent &> &Window::onKeyUpEvent() {
    return _onKeyUpEvent;
}

Event<Window *, const PointerEvent &> &Window::onPointerPressedEvent() {
    return _onPointerPressedEvent;
}

Event<Window *, const PointerEvent &> &Window::onPointerReleasedEvent() {
    return _onPointerReleasedEvent;
}

Event<Window *, const PointerEvent &> &Window::onPointerDraggedEvent() {
    return _onPointerDraggedEvent;
}

Event<Window *, const PointerEvent &> &Window::onPointerHoverEvent() {
    return _onPointerHoverEvent;
}

Event<Window *, const PointerEvent &> &Window::onMouseWheelEvent() {
    return _onMouseWheelEvent;
}

Event<Window *, bool> &Window::onPointerEnterEvent() {
    return _onPointerEnterEvent;
}

void Window::setRenderer(const RendererPtr &renderer) { _renderer = renderer; }

void Window::onUpdateEnabled(bool enabled) { UNUSED_VARIABLE(enabled); }

}  // namespace gfx
}  // namespace jet
