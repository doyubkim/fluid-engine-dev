// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_GL

#include <jet.viz/glfw_app.h>
#include <jet.viz/glfw_window.h>

#include <algorithm>
#include <cassert>
#include <vector>

using namespace jet;
using namespace viz;

std::vector<GLFWWindowPtr> sWindows;
GLFWWindowPtr sCurrentWindow;

static GLFWWindowPtr findWindow(GLFWwindow* glfwWindow) {
    for (auto w : sWindows) {
        if (w->glfwWindow() == glfwWindow) {
            return w;
        }
    }

    return nullptr;
}

int GLFWApp::initialize() {
    glfwSetErrorCallback(onErrorEvent);

    if (!glfwInit()) {
        return -1;
    }

    // Use OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef JET_MACOSX
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    return 0;
}

int GLFWApp::run() {
    while (sCurrentWindow != nullptr) {
        glfwWaitEvents();

        auto window = sCurrentWindow->glfwWindow();

        if (sCurrentWindow->isAnimationEnabled() ||
            sCurrentWindow->_renderRequested) {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            sCurrentWindow->resize(width, height);

            if (sCurrentWindow->isAnimationEnabled()) {
                sCurrentWindow->update();
            }

            sCurrentWindow->render();

            sCurrentWindow->_renderRequested = false;

            if (sCurrentWindow->isAnimationEnabled()) {
                glfwPostEmptyEvent();
            }
        }

        if (glfwWindowShouldClose(window)) {
            onCloseCurrentWindow(sCurrentWindow);
        }
    }

    glfwTerminate();

    return 0;
}

GLFWWindowPtr GLFWApp::createWindow(const std::string& title, int width,
                                    int height) {
    sCurrentWindow = GLFWWindowPtr(new GLFWWindow(title, width, height));
    sWindows.push_back(sCurrentWindow);

    auto glfwWindow = sCurrentWindow->glfwWindow();

#ifdef JET_USE_GLEW
    // Per-window operations
    glewExperimental = GL_TRUE;
    glewInit();

    if (GLEW_VERSION_3_2 == GL_FALSE) {
        printf(
            "This system doesn't support OpenGL 3.2. You may see nothing from "
            "this window.\n");
    }
#endif

    glfwSetKeyCallback(glfwWindow, onKeyEvent);
    glfwSetMouseButtonCallback(glfwWindow, onMouseButtonEvent);
    glfwSetCursorPosCallback(glfwWindow, onMouseCursorPosEvent);
    glfwSetCursorEnterCallback(glfwWindow, onMouseCursorEnterEvent);
    glfwSetScrollCallback(glfwWindow, onMouseScrollEvent);
    return sCurrentWindow;
}

void GLFWApp::onSetCurrentWindow(const GLFWWindowPtr& window) {
    assert(std::find(sWindows.begin(), sWindows.end(), window) !=
           sWindows.end());

    sCurrentWindow = window;
}

void GLFWApp::onCloseCurrentWindow(const GLFWWindowPtr& window) {
    auto it = std::find(sWindows.begin(), sWindows.end(), window);
    sWindows.erase(it);

    if (sCurrentWindow == window) {
        sCurrentWindow.reset();

        if (!sWindows.empty()) {
            sCurrentWindow = *sWindows.rbegin();
        }
    }
}

void GLFWApp::onKeyEvent(GLFWwindow* glfwWindow, int key, int scancode,
                         int action, int mods) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->key(key, scancode, action, mods);
}

void GLFWApp::onMouseButtonEvent(GLFWwindow* glfwWindow, int button, int action,
                                 int mods) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->pointerButton(button, action, mods);
}

void GLFWApp::onMouseCursorEnterEvent(GLFWwindow* glfwWindow, int entered) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->pointerEnter(entered == GL_TRUE);
}

void GLFWApp::onMouseCursorPosEvent(GLFWwindow* glfwWindow, double x,
                                    double y) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->pointerMoved(x, y);
}

void GLFWApp::onMouseScrollEvent(GLFWwindow* glfwWindow, double deltaX,
                                 double deltaY) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->mouseWheel(deltaX, deltaY);
}

void GLFWApp::onErrorEvent(int error, const char* description) {
    JET_ERROR << "GLFW Error [" << error << "] " << description;
}

#endif  // JET_USE_GL
