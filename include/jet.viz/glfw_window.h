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

//!
//! \brief Helper class for GLFW-based window.
//!
//! \see GlfwApp
//!
class GlfwWindow final {
 public:
    //! Sets view controller to the window.
    void setViewController(const ViewControllerPtr& viewController);

    //! Returns OpenGL renderer.
    const GLRendererPtr& renderer() const;

    //! Returns raw GLFW window object.
    GLFWwindow* glfwWindow() const;

    //! Request to render given number of frames to the renderer.
    void requestRender(unsigned int numFrames = 1);

    //! Returns true if update is enabled.
    bool isUpdateEnabled() const;

    //! Enables/disables update.
    void setIsUpdateEnabled(bool enabled);

    //! Sets swap interval.
    void setSwapInterval(int interval);

    // Event handlers

    //!
    //! \brief Returns update event object.
    //!
    //! An update callback function can be attached to this event object such
    //! as:
    //!
    //! \code{.cpp}
    //! bool onUpdate(GlfwWindow* win) { ... }
    //! ...
    //! window->onUpdateEvent() += onUpdate;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*>& onUpdateEvent();

    //!
    //! \brief Returns GUI update event object.
    //!
    //! An ImGui update callback function can be attached to this event object
    //! such as:
    //!
    //! \code{.cpp}
    //! bool onGui(GlfwWindow*) {
    //!     ImGui_ImplGlfwGL3_NewFrame();
    //!     ImGui::Begin("Info");
    //!     ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
    //!                 1000.0f / ImGui::GetIO().Framerate,
    //!                 ImGui::GetIO().Framerate);
    //!     ImGui::End();
    //!     ImGui::Render();
    //!     return true;
    //! }
    //! ...
    //! window->onUpdateEvent() += onGui;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*>& onGuiEvent();

    //!
    //! \brief Returns key-down event object.
    //!
    //! A key-down callback function can be attached to this event object such
    //! as:
    //!
    //! \code{.cpp}
    //! bool onKeyDown(GlfwWindow* win, const KeyEvent& keyEvent) { ... }
    //! ...
    //! window->onKeyDownEvent() += onKeyDown;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*, const KeyEvent&>& onKeyDownEvent();

    //!
    //! \brief Returns key-up event object.
    //!
    //! A key-up callback function can be attached to this event object such
    //! as:
    //!
    //! \code{.cpp}
    //! bool onKeyUp(GlfwWindow* win, const KeyEvent& keyEvent) { ... }
    //! ...
    //! window->onKeyUpEvent() += onKeyUp;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*, const KeyEvent&>& onKeyUpEvent();

    //!
    //! \brief Returns pointer-pressed event object.
    //!
    //! A pointer-pressed callback function can be attached to this event object
    //! such as:
    //!
    //! \code{.cpp}
    //! bool onPointerPressed(GlfwWindow* win, const PointerEvent& pointerEvent)
    //! { ... }
    //! ...
    //! window->onPointerPressedEvent() += onPointerPressed;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*, const PointerEvent&>& onPointerPressedEvent();

    //!
    //! \brief Returns pointer-released event object.
    //!
    //! A pointer-released callback function can be attached to this event
    //! object such as:
    //!
    //! \code{.cpp}
    //! bool onPointerReleased(GlfwWindow* win, const PointerEvent&
    //! pointerEvent) { ... }
    //! ...
    //! window->onPointerReleasedEvent() += onPointerReleased;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*, const PointerEvent&>& onPointerReleasedEvent();

    //!
    //! \brief Returns pointer-dragged event object.
    //!
    //! A pointer-dragged callback function can be attached to this event
    //! object such as:
    //!
    //! \code{.cpp}
    //! bool onPointerDragged(GlfwWindow* win, const PointerEvent&
    //! pointerEvent) { ... }
    //! ...
    //! window->onPointerDraggedEvent() += onPointerDragged;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*, const PointerEvent&>& onPointerDraggedEvent();

    //!
    //! \brief Returns pointer-hover event object.
    //!
    //! A pointer-hover (pointer button is in released state, but the pointer is
    //! moving) callback function can be attached to this event object such as:
    //!
    //! \code{.cpp}
    //! bool onPointerHover(GlfwWindow* win, const PointerEvent&
    //! pointerEvent) { ... }
    //! ...
    //! window->onPointerHoverEvent() += onPointerHover;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*, const PointerEvent&>& onPointerHoverEvent();

    //!
    //! \brief Returns mouse wheel event object.
    //!
    //! A mouse wheel callback function can be attached to this event
    //! object such as:
    //!
    //! \code{.cpp}
    //! bool onMouseWheel(GlfwWindow* win, const PointerEvent&
    //! pointerEvent) { ... }
    //! ...
    //! window->onMouseWheelEvent() += onMouseWheel;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<GlfwWindow*, const PointerEvent&>& onMouseWheelEvent();

    //! Returns the framebuffer size.
    //! Note that the framebuffer size can be different from the window size,
    //! especially on a Retina display (2x the window size).
    Size2 framebufferSize() const;

    //! Returns the window size.
    Size2 windowSize() const;

 private:
    GLFWwindow* _window = nullptr;

    MouseButtonType _pressedMouseButton = MouseButtonType::None;
    ModifierKey _lastModifierKey = ModifierKey::None;

    bool _isUpdateEnabled = false;
    unsigned int _numRequestedRenderFrames = 0;

    int _width = 1;
    int _height = 1;
    bool _hasPointerEntered = false;
    double _pointerPosX = 0.0;
    double _pointerPosY = 0.0;
    double _pointerDeltaX = 0.0;
    double _pointerDeltaY = 0.0;

    GLRendererPtr _renderer;
    ViewControllerPtr _viewController;

    int _swapInterval = 0;

    Event<GlfwWindow*> _onUpdateEvent;
    Event<GlfwWindow*> _onGuiEvent;
    Event<GlfwWindow*, const KeyEvent&> _onKeyDownEvent;
    Event<GlfwWindow*, const KeyEvent&> _onKeyUpEvent;
    Event<GlfwWindow*, const PointerEvent&> _onPointerPressedEvent;
    Event<GlfwWindow*, const PointerEvent&> _onPointerReleasedEvent;
    Event<GlfwWindow*, const PointerEvent&> _onPointerDraggedEvent;
    Event<GlfwWindow*, const PointerEvent&> _onPointerHoverEvent;
    Event<GlfwWindow*, const PointerEvent&> _onMouseWheelEvent;

    EventToken _onBasicCameraStateChangedEventToken = kEmptyEventToken;

    GlfwWindow(const std::string& title, int width, int height);

    void render();

    void resize(int width, int height);

    void update();

    void key(int key, int scancode, int action, int mods);

    void pointerButton(int button, int action, int mods);

    void pointerMoved(double x, double y);

    void pointerEnter(bool entered);

    void mouseWheel(double deltaX, double deltaY);

    double getScaleFactor() const;

    friend class GlfwApp;
};

typedef std::shared_ptr<GlfwWindow> GlfwWindowPtr;

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GLFW_WINDOW_H_
