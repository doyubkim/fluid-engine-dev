// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_WINDOW_H_
#define INCLUDE_JET_GFX_WINDOW_H_

#include <jet.gfx/event.h>
#include <jet.gfx/input_events.h>
#include <jet.gfx/renderer.h>
#include <jet.gfx/view_controller.h>

namespace jet {
namespace gfx {

//!
//! \brief Abstract base class for window.
//!
class Window {
 public:
    Window() = default;

    virtual ~Window() = default;

    //! Returns the framebuffer size.
    //! Note that the framebuffer size can be different from the window size,
    //! especially on a Retina display (2x the window size).
    virtual Vector2UZ framebufferSize() const = 0;

    //! Returns the window size.
    virtual Vector2UZ windowSize() const = 0;

    //! Request to render given number of frames to the renderer.
    virtual void requestRender(unsigned int numFrames);

    //! Sets swap interval.
    virtual void setSwapInterval(int interval);

    //! Returns current view controller of the window.
    const ViewControllerPtr& viewController() const;

    //! Sets view controller to the window.
    void setViewController(const ViewControllerPtr& viewController);

    //! Returns renderer.
    const RendererPtr& renderer() const;

    //! Returns true if update is enabled.
    bool isUpdateEnabled() const;

    //! Enables/disables update.
    void setIsUpdateEnabled(bool enabled);

    // Event handlers

    //!
    //! \brief Returns update event object.
    //!
    //! An update callback function can be attached to this event object such
    //! as:
    //!
    //! \code{.cpp}
    //! bool onUpdate(Window* win) { ... }
    //! ...
    //! window->onUpdateEvent() += onUpdate;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*>& onUpdateEvent();

    //!
    //! \brief Returns GUI update event object.
    //!
    //! An ImGui update callback function can be attached to this event object.
    //! For example with GlfwWindow:
    //!
    //! \code{.cpp}
    //! bool onGui(Window*) {
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
    Event<Window*>& onGuiEvent();

    //!
    //! \brief Returns window resized event object.
    //!
    //! A key-down callback function can be attached to this event object such
    //! as:
    //!
    //! \code{.cpp}
    //! bool onWindowResized(Window* win, const Vector2I& newSize) { ... }
    //! ...
    //! window->onWindowResizedEvent() += onWindowResized;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*, const Vector2I&>& onWindowResizedEvent();

    //!
    //! \brief Returns key-down event object.
    //!
    //! A key-down callback function can be attached to this event object such
    //! as:
    //!
    //! \code{.cpp}
    //! bool onKeyDown(Window* win, const KeyEvent& keyEvent) { ... }
    //! ...
    //! window->onKeyDownEvent() += onKeyDown;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*, const KeyEvent&>& onKeyDownEvent();

    //!
    //! \brief Returns key-up event object.
    //!
    //! A key-up callback function can be attached to this event object such
    //! as:
    //!
    //! \code{.cpp}
    //! bool onKeyUp(Window* win, const KeyEvent& keyEvent) { ... }
    //! ...
    //! window->onKeyUpEvent() += onKeyUp;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*, const KeyEvent&>& onKeyUpEvent();

    //!
    //! \brief Returns pointer-pressed event object.
    //!
    //! A pointer-pressed callback function can be attached to this event object
    //! such as:
    //!
    //! \code{.cpp}
    //! bool onPointerPressed(Window* win, const PointerEvent& pointerEvent)
    //! { ... }
    //! ...
    //! window->onPointerPressedEvent() += onPointerPressed;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*, const PointerEvent&>& onPointerPressedEvent();

    //!
    //! \brief Returns pointer-released event object.
    //!
    //! A pointer-released callback function can be attached to this event
    //! object such as:
    //!
    //! \code{.cpp}
    //! bool onPointerReleased(Window* win, const PointerEvent&
    //! pointerEvent) { ... }
    //! ...
    //! window->onPointerReleasedEvent() += onPointerReleased;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*, const PointerEvent&>& onPointerReleasedEvent();

    //!
    //! \brief Returns pointer-dragged event object.
    //!
    //! A pointer-dragged callback function can be attached to this event
    //! object such as:
    //!
    //! \code{.cpp}
    //! bool onPointerDragged(Window* win, const PointerEvent&
    //! pointerEvent) { ... }
    //! ...
    //! window->onPointerDraggedEvent() += onPointerDragged;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*, const PointerEvent&>& onPointerDraggedEvent();

    //!
    //! \brief Returns pointer-hover event object.
    //!
    //! A pointer-hover (pointer button is in released state, but the pointer is
    //! moving) callback function can be attached to this event object such as:
    //!
    //! \code{.cpp}
    //! bool onPointerHover(Window* win, const PointerEvent&
    //! pointerEvent) { ... }
    //! ...
    //! window->onPointerHoverEvent() += onPointerHover;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*, const PointerEvent&>& onPointerHoverEvent();

    //!
    //! \brief Returns mouse wheel event object.
    //!
    //! A mouse wheel callback function can be attached to this event
    //! object such as:
    //!
    //! \code{.cpp}
    //! bool onMouseWheel(Window* win, const PointerEvent&
    //! pointerEvent) { ... }
    //! ...
    //! window->onMouseWheelEvent() += onMouseWheel;
    //! \endcode
    //!
    //! \return Event object.
    //!
    Event<Window*, const PointerEvent&>& onMouseWheelEvent();

 protected:
    //! Sets the renderer for this window.
    void setRenderer(const RendererPtr& renderer);

 private:
    bool _isUpdateEnabled = false;

    RendererPtr _renderer;
    ViewControllerPtr _viewController;

    Event<Window*> _onUpdateEvent;
    Event<Window*> _onGuiEvent;
    Event<Window*, const Vector2I&> _onWindowResizedEvent;
    Event<Window*, const KeyEvent&> _onKeyDownEvent;
    Event<Window*, const KeyEvent&> _onKeyUpEvent;
    Event<Window*, const PointerEvent&> _onPointerPressedEvent;
    Event<Window*, const PointerEvent&> _onPointerReleasedEvent;
    Event<Window*, const PointerEvent&> _onPointerDraggedEvent;
    Event<Window*, const PointerEvent&> _onPointerHoverEvent;
    Event<Window*, const PointerEvent&> _onMouseWheelEvent;

    EventToken _onCameraStateChangedEventToken = kEmptyEventToken;
};

//! Shared typed for Window.
using WindowPtr = std::shared_ptr<Window>;

}  // namespace gfx
}  // namespace jet

#endif  // INCLUDE_JET_GFX_WINDOW_H_
