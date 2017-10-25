// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GLFW_APP_H_
#define INCLUDE_JET_VIZ_GLFW_APP_H_

#ifdef JET_USE_GL

#include <jet/macros.h>

#include <memory>
#include <string>

struct GLFWwindow;

namespace jet { namespace viz {

class GLFWWindow;
typedef std::shared_ptr<GLFWWindow> GLFWWindowPtr;

class GLFWApp {
 public:
    static int initialize();

    static int run();

    static GLFWWindowPtr createWindow(const std::string& title, int width,
                                      int height);

    friend class GLFWWindow;

 private:
    static void onSetCurrentWindow(const GLFWWindowPtr& window);

    static void onCloseCurrentWindow(const GLFWWindowPtr& window);

    static void onKeyEvent(GLFWwindow* glfwWindow, int key, int scancode,
                           int action, int mods);

    static void onMouseButtonEvent(GLFWwindow* glfwWindow, int button,
                                   int action, int mods);

    static void onMouseCursorEnterEvent(GLFWwindow* glfwWindow, int entered);

    static void onMouseCursorPosEvent(GLFWwindow* glfwWindow, double x,
                                      double y);

    static void onMouseScrollEvent(GLFWwindow* glfwWindow, double deltaX,
                                   double deltaY);

    static void onErrorEvent(int error, const char* description);
};

} }  // namespace jet::viz

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GLFW_APP_H_
