// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GLFW_WINDOW_H_
#define INCLUDE_JET_VIZ_GLFW_WINDOW_H_

#ifdef JET_USE_GL

#include <jet/macros.h>

#include <jet.viz/event.h>
#include <jet.viz/gl_renderer.h>
#include <jet.viz/glfw_app.h>
#include <jet.viz/input_events.h>
#include <jet.viz/view_controller.h>

struct GLFWwindow;

namespace jet {
namespace viz {

class GLFWWindow final {
 public:
    void setViewController(const ViewControllerPtr& viewController);

    const GLRendererPtr& renderer() const;

    GLFWwindow* glfwWindow() const;

    void requestRender();

    bool isAnimationEnabled() const;
    void setIsAnimationEnabled(bool enabled);

    // Event handlers
    Event<GLFWWindow*>& onUpdateEvent();
    Event<GLFWWindow*, const KeyEvent&>& onKeyDownEvent();
    Event<GLFWWindow*, const KeyEvent&>& onKeyUpEvent();
    Event<GLFWWindow*, const PointerEvent&>& onPointerPressedEvent();
    Event<GLFWWindow*, const PointerEvent&>& onPointerReleasedEvent();
    Event<GLFWWindow*, const PointerEvent&>& onPointerDraggedEvent();
    Event<GLFWWindow*, const PointerEvent&>& onPointerHoverEvent();
    Event<GLFWWindow*, const PointerEvent&>& onMouseWheelEvent();

 private:
    GLFWwindow* _window = nullptr;

    MouseButtonType _pressedMouseButton = MouseButtonType::None;
    ModifierKey _lastModifierKey = ModifierKey::None;

    bool _isAnimationEnabled = false;
    bool _renderRequested = false;

    int _width = 1;
    int _height = 1;
    bool _hasPointerEntered = false;
    double _pointerPosX = 0.0;
    double _pointerPosY = 0.0;
    double _pointerDeltaX = 0.0;
    double _pointerDeltaY = 0.0;

    GLRendererPtr _renderer;
    ViewControllerPtr _viewController;

    Event<GLFWWindow*> _onUpdateEvent;
    Event<GLFWWindow*, const KeyEvent&> _onKeyDownEvent;
    Event<GLFWWindow*, const KeyEvent&> _onKeyUpEvent;
    Event<GLFWWindow*, const PointerEvent&> _onPointerPressedEvent;
    Event<GLFWWindow*, const PointerEvent&> _onPointerReleasedEvent;
    Event<GLFWWindow*, const PointerEvent&> _onPointerDraggedEvent;
    Event<GLFWWindow*, const PointerEvent&> _onPointerHoverEvent;
    Event<GLFWWindow*, const PointerEvent&> _onMouseWheelEvent;

    EventToken _onBasicCameraStateChangedEventToken = kEmptyEventToken;

    GLFWWindow(const std::string& title, int width, int height);

    void render();

    void resize(int width, int height);

    void update();

    void key(int key, int scancode, int action, int mods);

    void pointerButton(int button, int action, int mods);

    void pointerMoved(double x, double y);

    void pointerEnter(bool entered);

    void mouseWheel(double deltaX, double deltaY);

    double getScaleFactor() const;

    friend class GLFWApp;
};

typedef std::shared_ptr<GLFWWindow> GLFWWindowPtr;

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GLFW_WINDOW_H_
