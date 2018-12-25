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
        std::make_shared<PerspCamera>()));

    resize(width, height);
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

void GlfwWindow::requestRender(unsigned int numFrames) {
    _numRequestedRenderFrames = std::max(_numRequestedRenderFrames, numFrames);
    glfwPostEmptyEvent();
}

GLFWwindow *GlfwWindow::glfwWindow() const { return _window; }

void GlfwWindow::render() {
    JET_ASSERT(renderer());

    renderer()->render();

    onGuiEvent()(this);
}

void GlfwWindow::resize(int width, int height) {
    JET_ASSERT(renderer());

    double scaleFactor = getScaleFactor();

    Viewport viewport;
    viewport.x = 0.0;
    viewport.y = 0.0;
    viewport.width = scaleFactor * width;
    viewport.height = scaleFactor * height;

    _width = width;
    _height = height;

    viewController()->setViewport(viewport);

    onWindowResizedEvent()(this, {width, height});
}

void GlfwWindow::update() {
    // Update
    onUpdateEvent()(this);
}

void GlfwWindow::key(int key, int scancode, int action, int mods) {
    UNUSED_VARIABLE(scancode);

    ModifierKey modifier = getModifier(mods);
    _lastModifierKey = modifier;

    KeyEvent keyEvent(key, modifier);

    if (action == GLFW_PRESS) {
        if (viewController() != nullptr) {
            viewController()->keyDown(keyEvent);
        }
        onKeyDownEvent()(this, keyEvent);
    } else if (action == GLFW_RELEASE) {
        if (viewController() != nullptr) {
            viewController()->keyUp(keyEvent);
        }
        onKeyUpEvent()(this, keyEvent);
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
        if (viewController() != nullptr) {
            viewController()->pointerPressed(pointerEvent);
            onPointerPressedEvent()(this, pointerEvent);
        }
    } else if (action == GLFW_RELEASE) {
        if (viewController() != nullptr) {
            viewController()->pointerReleased(pointerEvent);
            onPointerReleasedEvent()(this, pointerEvent);
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
        if (viewController() != nullptr) {
            viewController()->pointerDragged(pointerEvent);
        }
        onPointerDraggedEvent()(this, pointerEvent);
    } else {
        if (viewController() != nullptr) {
            viewController()->pointerHover(pointerEvent);
        }
        onPointerHoverEvent()(this, pointerEvent);
    }
}

void GlfwWindow::mouseWheel(double deltaX, double deltaY) {
    MouseWheelData wheelData;
    wheelData.deltaX = deltaX;
    wheelData.deltaY = deltaY;

    PointerEvent pointerEvent(PointerInputType::kMouse, _lastModifierKey,
                              _pointerPosX, _pointerPosY, _pointerDeltaX,
                              _pointerDeltaY, _pressedMouseButton, wheelData);

    if (viewController() != nullptr) {
        viewController()->mouseWheel(pointerEvent);
    }
    onMouseWheelEvent()(this, pointerEvent);
}

void GlfwWindow::pointerEnter(bool entered) { _hasPointerEntered = entered; }

double GlfwWindow::getScaleFactor() const {
    int fbWidth, fbHeight;
    int winWidth, winHeight;
    glfwGetFramebufferSize(_window, &fbWidth, &fbHeight);
    glfwGetWindowSize(_window, &winWidth, &winHeight);

    return static_cast<float>(fbWidth) / static_cast<float>(winWidth);
}

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL
