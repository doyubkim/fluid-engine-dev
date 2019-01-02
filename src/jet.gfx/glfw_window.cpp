// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#ifdef JET_USE_GL

#include <jet.gfx/gl_common.h>
#include <jet.gfx/glfw_window.h>
#include <jet.gfx/persp_camera.h>
#include <jet.gfx/pitch_yaw_view_controller.h>

namespace jet {
namespace gfx {

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

GlfwWindow::GlfwWindow(const std::string &title, int width, int height) {
    _window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

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

    setRenderer(std::make_shared<GLRenderer>());

    setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>(), Vector3F()));

    onWindowResized(width, height);
}

void GlfwWindow::setSwapInterval(int interval) {
    _swapInterval = interval;
    glfwSwapInterval(interval);
}

Vector2UZ GlfwWindow::framebufferSize() const {
    int w, h;
    glfwGetFramebufferSize(_window, &w, &h);
    return Vector2UZ(static_cast<size_t>(w), static_cast<size_t>(h));
}

Vector2UZ GlfwWindow::windowSize() const {
    int w, h;
    glfwGetWindowSize(_window, &w, &h);
    return Vector2UZ(static_cast<size_t>(w), static_cast<size_t>(h));
}

Vector2F GlfwWindow::displayScalingFactor() const {
    if (!_hasDisplayScalingFactorCache) {
        _displayScalingFactorCache = Window::displayScalingFactor();
        _hasDisplayScalingFactorCache = true;
    }
    return _displayScalingFactorCache;
}

void GlfwWindow::requestRender(unsigned int numFrames) {
    _numRequestedRenderFrames = std::max(_numRequestedRenderFrames, numFrames);
    glfwPostEmptyEvent();
}

GLFWwindow *GlfwWindow::glfwWindow() const { return _window; }

void GlfwWindow::onRender() {
    JET_ASSERT(renderer());

    renderer()->render();

    onGuiEvent()(this);
}

bool GlfwWindow::onWindowResized(int width, int height) {
    JET_ASSERT(renderer());

    Vector2F scaleFactor = displayScalingFactor();

    Viewport viewport;
    viewport.x = 0.0;
    viewport.y = 0.0;
    viewport.width = scaleFactor.x * width;
    viewport.height = scaleFactor.y * height;

    _width = width;
    _height = height;

    viewController()->setViewport(viewport);

    return onWindowResizedEvent()(this, {width, height});
}

bool GlfwWindow::onWindowMoved(int x, int y) {
    // In case the window has moved to a different monitor with different DPI
    // setting.
    _hasDisplayScalingFactorCache = false;

    return onWindowMovedEvent()(this, {x, y});
}

bool GlfwWindow::onUpdate() {
    // Update
    return onUpdateEvent()(this);
}

bool GlfwWindow::onKey(int key, int scancode, int action, int mods) {
    UNUSED_VARIABLE(scancode);

    ModifierKey modifier = getModifier(mods);
    _lastModifierKey = modifier;

    KeyEvent keyEvent(key, modifier);

    if (action == GLFW_PRESS) {
        if (viewController() != nullptr) {
            viewController()->keyDown(keyEvent);
        }
        return onKeyDownEvent()(this, keyEvent);
    } else if (action == GLFW_RELEASE) {
        if (viewController() != nullptr) {
            viewController()->keyUp(keyEvent);
        }
        return onKeyUpEvent()(this, keyEvent);
    }

    return false;
}

bool GlfwWindow::onPointerButton(int button, int action, int mods) {
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
        if (viewController() != nullptr) {
            viewController()->pointerPressed(pointerEvent);
            return onPointerPressedEvent()(this, pointerEvent);
        }
    } else if (action == GLFW_RELEASE) {
        _pressedMouseButton = MouseButtonType::kNone;

        if (viewController() != nullptr) {
            viewController()->pointerReleased(pointerEvent);
            return onPointerReleasedEvent()(this, pointerEvent);
        }
    }

    return false;
}

bool GlfwWindow::onPointerMoved(double x, double y) {
    Vector2F scaleFactor = displayScalingFactor();
    x = scaleFactor.x * x;
    y = scaleFactor.y * y;

    _pointerDeltaX = x - _pointerPosX;
    _pointerDeltaY = y - _pointerPosY;

    _pointerPosX = x;
    _pointerPosY = y;

    PointerEvent pointerEvent(
        PointerInputType::kMouse, _lastModifierKey, _pointerPosX, _pointerPosY,
        _pointerDeltaX, _pointerDeltaY, _pressedMouseButton, MouseWheelData());

    if (_pressedMouseButton != MouseButtonType::kNone) {
        if (viewController() != nullptr) {
            viewController()->pointerDragged(pointerEvent);
        }
        return onPointerDraggedEvent()(this, pointerEvent);
    } else {
        if (viewController() != nullptr) {
            viewController()->pointerHover(pointerEvent);
        }
        return onPointerHoverEvent()(this, pointerEvent);
    }
}

bool GlfwWindow::onMouseWheel(double deltaX, double deltaY) {
    MouseWheelData wheelData;
    wheelData.deltaX = deltaX;
    wheelData.deltaY = deltaY;

    PointerEvent pointerEvent(PointerInputType::kMouse, _lastModifierKey,
                              _pointerPosX, _pointerPosY, _pointerDeltaX,
                              _pointerDeltaY, _pressedMouseButton, wheelData);

    if (viewController() != nullptr) {
        viewController()->mouseWheel(pointerEvent);
    }
    return onMouseWheelEvent()(this, pointerEvent);
}

bool GlfwWindow::onPointerEnter(bool entered) {
    _hasPointerEntered = entered;

    return onPointerEnterEvent()(this);
}

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL
