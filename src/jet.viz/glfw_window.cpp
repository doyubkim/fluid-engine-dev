// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_GL

#include <jet.viz/glfw_window.h>

using namespace jet;
using namespace viz;

static ModifierKey getModifier(int mods) {
    ModifierKey modifier = ModifierKey::kNone;

    if (mods == GLFW_MOD_ALT) {
        modifier = ModifierKey::kAlt;
    } else if (mods == GLFW_MOD_CONTROL) {
        modifier = ModifierKey::kCtrl;
    } else if (mods == GLFW_MOD_SHIFT) {
        modifier = ModifierKey::kShift;
    }

    return modifier;
}

GlfwWindow::GlfwWindow(const std::string& title, int width, int height) {
    _window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);

    glfwMakeContextCurrent(_window);
    setSwapInterval(_swapInterval);

    if (gl3wInit()) {
        JET_ERROR << "failed to initialize OpenGL";
        return;
    }

    if (!gl3wIsSupported(3, 3)) {
        JET_ERROR << "OpenGL 3.3 not supported";
        return;
    }

    JET_INFO << "OpenGL " << glGetString(GL_VERSION) << " GLSL "
             << glGetString(GL_SHADING_LANGUAGE_VERSION);

    _renderer = std::make_shared<GLRenderer>();
}

void GlfwWindow::setViewController(const ViewControllerPtr& viewController) {
    // Detach event handler from old view controller
    if (_onBasicCameraStateChangedEventToken != kEmptyEventToken &&
        _viewController != nullptr) {
        _viewController->onBasicCameraStateChanged() -=
            _onBasicCameraStateChangedEventToken;
    }

    _viewController = viewController;

    _onBasicCameraStateChangedEventToken =
        viewController->onBasicCameraStateChanged() += [this](ViewController*) {
            // When basic camera state changes, update the view.
            requestRender();
            return true;
        };

    _renderer->setCamera(_viewController->camera());
}

const GLRendererPtr& GlfwWindow::renderer() const { return _renderer; }

bool GlfwWindow::isUpdateEnabled() const { return _isUpdateEnabled; }

void GlfwWindow::setIsUpdateEnabled(bool enabled) {
    _isUpdateEnabled = enabled;
}

void GlfwWindow::setSwapInterval(int interval) {
    _swapInterval = interval;
    glfwSwapInterval(interval);
}

GLFWwindow* GlfwWindow::glfwWindow() const { return _window; }

void GlfwWindow::requestRender(unsigned int numFrames) {
    _numRequestedRenderFrames = std::max(_numRequestedRenderFrames, numFrames);
    glfwPostEmptyEvent();
}

Event<GlfwWindow*>& GlfwWindow::onUpdateEvent() { return _onUpdateEvent; }

Event<GlfwWindow*>& GlfwWindow::onGuiEvent() { return _onGuiEvent; }

Event<GlfwWindow*, const KeyEvent&>& GlfwWindow::onKeyDownEvent() {
    return _onKeyDownEvent;
}

Event<GlfwWindow*, const KeyEvent&>& GlfwWindow::onKeyUpEvent() {
    return _onKeyUpEvent;
}

Event<GlfwWindow*, const PointerEvent&>& GlfwWindow::onPointerPressedEvent() {
    return _onPointerPressedEvent;
}

Event<GlfwWindow*, const PointerEvent&>& GlfwWindow::onPointerReleasedEvent() {
    return _onPointerReleasedEvent;
}

Event<GlfwWindow*, const PointerEvent&>& GlfwWindow::onPointerDraggedEvent() {
    return _onPointerDraggedEvent;
}

Event<GlfwWindow*, const PointerEvent&>& GlfwWindow::onPointerHoverEvent() {
    return _onPointerHoverEvent;
}

Event<GlfwWindow*, const PointerEvent&>& GlfwWindow::onMouseWheelEvent() {
    return _onMouseWheelEvent;
}

Size2 GlfwWindow::framebufferSize() const {
    int w, h;
    glfwGetFramebufferSize(_window, &w, &h);
    return Size2{static_cast<size_t>(w), static_cast<size_t>(h)};
}

Size2 GlfwWindow::windowSize() const {
    int w, h;
    glfwGetWindowSize(_window, &w, &h);
    return Size2{static_cast<size_t>(w), static_cast<size_t>(h)};
}

void GlfwWindow::render() {
    if (_renderer != nullptr) {
        _renderer->render();
    }

    _onGuiEvent(this);
}

void GlfwWindow::resize(int width, int height) {
    if (_renderer != nullptr) {
        Viewport viewport;
        viewport.x = 0.0;
        viewport.y = 0.0;
        viewport.width = width;
        viewport.height = height;

        _width = width;
        _height = height;

        _viewController->resize(viewport);
        _renderer->resize(viewport);
    }
}

void GlfwWindow::update() {
    // Update
    _onUpdateEvent(this);
}

void GlfwWindow::key(int key, int scancode, int action, int mods) {
    UNUSED_VARIABLE(scancode);

    ModifierKey modifier = getModifier(mods);
    _lastModifierKey = modifier;

    KeyEvent keyEvent(key, modifier);

    if (action == GLFW_PRESS) {
        if (_viewController != nullptr) {
            _viewController->keyDown(keyEvent);
        }
        _onKeyDownEvent(this, keyEvent);
    } else if (action == GLFW_RELEASE) {
        if (_viewController != nullptr) {
            _viewController->keyUp(keyEvent);
        }
        _onKeyUpEvent(this, keyEvent);
    }
}

void GlfwWindow::pointerButton(int button, int action, int mods) {
    PointerInputType newInputType = PointerInputType::kMouse;
    ModifierKey newModifierKey = getModifier(mods);

    _lastModifierKey = newModifierKey;

    _pressedMouseButton = MouseButtonType::kNone;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        _pressedMouseButton = MouseButtonType::kLeft;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        _pressedMouseButton = MouseButtonType::kRight;
    } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        _pressedMouseButton = MouseButtonType::kMiddle;
    }

    PointerEvent pointerEvent(newInputType, newModifierKey, _pointerPosX,
                              _pointerPosY, _pointerDeltaX, _pointerDeltaY,
                              _pressedMouseButton, MouseWheelData());

    if (action == GLFW_PRESS) {
        if (_viewController != nullptr) {
            _viewController->pointerPressed(pointerEvent);
            _onPointerPressedEvent(this, pointerEvent);
        }
    } else if (action == GLFW_RELEASE) {
        if (_viewController != nullptr) {
            _viewController->pointerReleased(pointerEvent);
            _onPointerReleasedEvent(this, pointerEvent);
        }

        _pressedMouseButton = MouseButtonType::kNone;
    }
}

void GlfwWindow::pointerMoved(double x, double y) {
    x = getScaleFactor() * x;
    y = getScaleFactor() * y;

    _pointerDeltaX = x - _pointerPosX;
    _pointerDeltaY = y - _pointerPosY;

    _pointerPosX = x;
    _pointerPosY = y;

    PointerEvent pointerEvent(
        PointerInputType::kMouse, _lastModifierKey, _pointerPosX, _pointerPosY,
        _pointerDeltaX, _pointerDeltaY, _pressedMouseButton, MouseWheelData());

    if (_pressedMouseButton != MouseButtonType::kNone) {
        if (_viewController != nullptr) {
            _viewController->pointerDragged(pointerEvent);
        }
        _onPointerDraggedEvent(this, pointerEvent);
    } else {
        if (_viewController != nullptr) {
            _viewController->pointerHover(pointerEvent);
        }
        _onPointerHoverEvent(this, pointerEvent);
    }
}

void GlfwWindow::mouseWheel(double deltaX, double deltaY) {
    MouseWheelData wheelData;
    wheelData.deltaX = deltaX;
    wheelData.deltaY = deltaY;

    PointerEvent pointerEvent(PointerInputType::kMouse, _lastModifierKey,
                              _pointerPosX, _pointerPosY, _pointerDeltaX,
                              _pointerDeltaY, _pressedMouseButton, wheelData);

    if (_viewController != nullptr) {
        _viewController->mouseWheel(pointerEvent);
    }
    _onMouseWheelEvent(this, pointerEvent);
}

void GlfwWindow::pointerEnter(bool entered) { _hasPointerEntered = entered; }

double GlfwWindow::getScaleFactor() const {
    int fbWidth, fbHeight;
    int winWidth, winHeight;
    glfwGetFramebufferSize(_window, &fbWidth, &fbHeight);
    glfwGetWindowSize(_window, &winWidth, &winHeight);

    return static_cast<float>(fbWidth) / static_cast<float>(winWidth);
}

#endif  // JET_USE_GL
