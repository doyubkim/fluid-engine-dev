// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GLFW_APP_H_
#define INCLUDE_JET_VIZ_GLFW_APP_H_

#ifdef JET_USE_GL

#include <jet/macros.h>

#include <jet.viz/event.h>

#include <memory>
#include <string>

struct GLFWwindow;

namespace jet {
namespace viz {

class GlfwWindow;
typedef std::shared_ptr<GlfwWindow> GlfwWindowPtr;

//!
//! \brief Helper class for GLFW-based applications.
//!
//! This class provides simple C++ wrapper around GLFW library. Here's a minimal
//! example that shows how to create and launch an OpenGL app:
//!
//! \code{.cpp}
//! #include <jet.viz/jet.viz.h>
//!
//! int main() {
//!     GlfwApp::initialize();
//!     auto window = GlfwApp::createWindow("OpenGL Tests", 1280, 720);
//!     GlfwApp::run();
//! }
//!
//! \endcode
//!
class GlfwApp {
 public:
    //! Initializes the app.
    static int initialize();

    //! Starts the run-loop.
    static int run();

    //!
    //! Creates a GLFW window.
    //!
    //! \param title Title of the window.
    //! \param width Width of the window.
    //! \param height Height of the window.
    //! \return GLFW Window object.
    //!
    static GlfwWindowPtr createWindow(const std::string& title, int width,
                                      int height);

    //!
    //! Finds Jet.Viz GLFW window object using raw GLFW window object.
    //!
    //! \param glfwWindow raw GLFW window object.
    //! \return Jet.Viz GLFW window object.
    //!
    static GlfwWindowPtr findWindow(GLFWwindow* glfwWindow);

    //!
    //! \brief Returns key event object.
    //!
    //! This function exposes low-level raw GLFW event handler. The callback
    //! function should be identical to GLFWkeyfun in GLFW. Unless there is a
    //! need for accessing raw GLFW event handling cycle, use onKeyDownEvent or
    //! onKeyUpEvent from GlfwWindow.
    //!
    //! Here's an example for attaching a callback function to the event object:
    //!
    //! \code{.cpp}
    //! GlfwApp::onGlfwKeyEvent() += callbackFunc;
    //! \endcode
    //!
    //! \return Event object.
    //!
    static Event<GLFWwindow*, int, int, int, int>& onGlfwKeyEvent();

    //!
    //! \brief Returns mouse button event object.
    //!
    //! This function exposes low-level raw GLFW event handler. The callback
    //! function should be identical to GLFWmousebuttonfun in GLFW. Unless there
    //! is a need for accessing raw GLFW event handling cycle, use
    //! onPointerPressedEvent or onPointerReleasedEvent from GlfwWindow.
    //!
    //! Here's an example for attaching a callback function to the event object:
    //!
    //! \code{.cpp}
    //! GlfwApp::onGlfwMouseButtonEvent() += callbackFunc;
    //! \endcode
    //!
    //! \return Event object.
    //!
    static Event<GLFWwindow*, int, int, int>& onGlfwMouseButtonEvent();

    //!
    //! \brief Returns mouse cursor position event object.
    //!
    //! This function exposes low-level raw GLFW event handler. The callback
    //! function should be identical to GLFWcursorposfun in GLFW. Unless there
    //! is a need for accessing raw GLFW event handling cycle, use
    //! onPointerDraggedEvent or onPointerHoverEvent from GlfwWindow.
    //!
    //! Here's an example for attaching a callback function to the event object:
    //!
    //! \code{.cpp}
    //! GlfwApp::onGlfwMouseCursorPosEvent() += callbackFunc;
    //! \endcode
    //!
    //! \return Event object.
    //!
    static Event<GLFWwindow*, double, double>& onGlfwMouseCursorPosEvent();

    //!
    //! \brief Returns mouse cursor enter event object.
    //!
    //! This function exposes low-level raw GLFW event handler. The callback
    //! function should be identical to GLFWcursorenterfun in GLFW. Here's an
    //! example for attaching a callback function to the event object:
    //!
    //! \code{.cpp}
    //! GlfwApp::onGlfwMouseCursorEnterEvent() += callbackFunc;
    //! \endcode
    //!
    //! \return Event object.
    //!
    static Event<GLFWwindow*, int>& onGlfwMouseCursorEnterEvent();

    //!
    //! \brief Returns mouse scroll event object.
    //!
    //! This function exposes low-level raw GLFW event handler. The callback
    //! function should be identical to  GLFWscrollfun in GLFW. Unless there
    //! is a need for accessing raw GLFW event handling cycle, use
    //! onMouseWheelEvent from GlfwWindow.
    //!
    //! Here's an example for attaching a callback function to the event object:
    //!
    //! \code{.cpp}
    //! GlfwApp::onGlfwMouseScrollEvent() += callbackFunc;
    //! \endcode
    //!
    //! \return Event object.
    //!
    static Event<GLFWwindow*, double, double>& onGlfwMouseScrollEvent();

    //!
    //! \brief Returns Unicode character event object.
    //!
    //! This function exposes low-level raw GLFW event handler. The callback
    //! function should be identical to GLFWcharfun in GLFW. Here's an
    //! example for attaching a callback function to the event object:
    //!
    //! \code{.cpp}
    //! GlfwApp::onGlfwCharEvent() += callbackFunc;
    //! \endcode
    //!
    //! \return Event object.
    //!
    static Event<GLFWwindow*, unsigned int>& onGlfwCharEvent();

    //!
    //! \brief Returns Unicode character with modifier event object.
    //!
    //! This function exposes low-level raw GLFW event handler. The callback
    //! function should be identical to GLFWcharmodsfun in GLFW. Here's an
    //! example for attaching a callback function to the event object:
    //!
    //! \code{.cpp}
    //! GlfwApp::onGlfwCharModsEvent() += callbackFunc;
    //! \endcode
    //!
    //! \return Event object.
    //!
    static Event<GLFWwindow*, unsigned int, int>& onGlfwCharModsEvent();

    //!
    //! \brief Returns Unicode character with modifier event object.
    //!
    //! This function exposes low-level raw GLFW event handler. The callback
    //! function should be identical to GLFWdropfun in GLFW. Here's an
    //! example for attaching a callback function to the event object:
    //!
    //! \code{.cpp}
    //! GlfwApp::onGlfwDropEvent() += callbackFunc;
    //! \endcode
    //!
    //! \return Event object.
    //!
    static Event<GLFWwindow*, int, const char**>& onGlfwDropEvent();

    friend class GlfwWindow;

 private:
    static void onSetCurrentWindow(const GlfwWindowPtr& window);

    static void onCloseCurrentWindow(const GlfwWindowPtr& window);

    static void onKey(GLFWwindow* glfwWindow, int key, int scancode, int action,
                      int mods);

    static void onMouseButton(GLFWwindow* glfwWindow, int button, int action,
                              int mods);

    static void onMouseCursorEnter(GLFWwindow* glfwWindow, int entered);

    static void onMouseCursorPos(GLFWwindow* glfwWindow, double x, double y);

    static void onMouseScroll(GLFWwindow* glfwWindow, double deltaX,
                              double deltaY);

    static void onChar(GLFWwindow* glfwWindow, unsigned int code);

    static void onCharMods(GLFWwindow* glfwWindow, unsigned int code, int mods);

    static void onDrop(GLFWwindow* glfwWindow, int numDroppedFiles,
                       const char** pathNames);

    static void onErrorEvent(int error, const char* description);
};

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GLFW_APP_H_
