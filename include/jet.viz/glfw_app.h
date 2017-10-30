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

class GLFWWindow;
typedef std::shared_ptr<GLFWWindow> GLFWWindowPtr;

class GLFWApp {
 public:
    static int initialize();

    static int run();

    static GLFWWindowPtr createWindow(const std::string& title, int width,
                                      int height);

    static GLFWWindowPtr findWindow(GLFWwindow* glfwWindow);

    static Event<GLFWwindow*, int, int, int, int>& onBeginGlfwKeyEvent();

    static Event<GLFWwindow*, int, int, int>& onBeginGlfwMouseButtonEvent();

    static Event<GLFWwindow*, double, double>& onBeginGlfwMouseCursorPosEvent();

    static Event<GLFWwindow*, int>& onBeginGlfwMouseCursorEnterEvent();

    static Event<GLFWwindow*, double, double>& onBeginGlfwMouseScrollEvent();

    static Event<GLFWwindow*, unsigned int>& onBeginGlfwCharEvent();

    static Event<GLFWwindow*, unsigned int, int>& onBeginGlfwCharModsEvent();

    static Event<GLFWwindow*, int, const char**>& onBeginGlfwDropEvent();

    friend class GLFWWindow;

 private:
    static void onSetCurrentWindow(const GLFWWindowPtr& window);

    static void onCloseCurrentWindow(const GLFWWindowPtr& window);

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
