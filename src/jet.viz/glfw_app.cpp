// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_GL

#include <jet.viz/glfw_app.h>
#include <jet.viz/glfw_window.h>

using namespace jet;
using namespace viz;

namespace {

std::vector<GLFWWindowPtr> sWindows;
GLFWWindowPtr sCurrentWindow;

Event<GLFWwindow*, int, int, int, int> sOnBeginGlfwKeyEvent;
Event<GLFWwindow*, int, int, int> sOnBeginGlfwMouseButtonEvent;
Event<GLFWwindow*, double, double> sOnBeginGlfwMouseCursorPosEvent;
Event<GLFWwindow*, int> sOnBeginGlfwMouseCursorEnterEvent;
Event<GLFWwindow*, double, double> sOnBeginGlfwMouseScrollEvent;
Event<GLFWwindow*, unsigned int> sOnBeginGlfwCharEvent;
Event<GLFWwindow*, unsigned int, int> sOnBeginGlfwCharModsEvent;
Event<GLFWwindow*, int, const char**> sOnBeginGlfwDropEvent;

}  // namespace

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
    // Force render first frame
    if (sCurrentWindow != nullptr) {
        sCurrentWindow->requestRender();
    }

    while (sCurrentWindow != nullptr) {
        JET_INFO << "blocked: " << sCurrentWindow->_renderRequested;
        glfwWaitEvents();
        JET_INFO << "escaped: " << sCurrentWindow->_renderRequested;

        auto window = sCurrentWindow->glfwWindow();

        if (sCurrentWindow->isAnimationEnabled() ||
            sCurrentWindow->_renderRequested) {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            sCurrentWindow->resize(width, height);

            if (sCurrentWindow->isAnimationEnabled()) {
                sCurrentWindow->update();
            }

            JET_INFO << "render";
            sCurrentWindow->render();

            // Reset render request
            sCurrentWindow->_renderRequested = false;

            if (sCurrentWindow->isAnimationEnabled()) {
                glfwPostEmptyEvent();
            }

            glfwSwapBuffers(sCurrentWindow->glfwWindow());
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

    glfwSetKeyCallback(glfwWindow, onKey);
    glfwSetMouseButtonCallback(glfwWindow, onMouseButton);
    glfwSetCursorPosCallback(glfwWindow, onMouseCursorPos);
    glfwSetCursorEnterCallback(glfwWindow, onMouseCursorEnter);
    glfwSetScrollCallback(glfwWindow, onMouseScroll);
    glfwSetCharCallback(glfwWindow, onChar);
    glfwSetCharModsCallback(glfwWindow, onCharMods);
    glfwSetDropCallback(glfwWindow, onDrop);

    return sCurrentWindow;
}

Event<GLFWwindow*, int, int, int, int>& GLFWApp::onBeginGlfwKeyEvent() {
    return sOnBeginGlfwKeyEvent;
}

Event<GLFWwindow*, int, int, int>& GLFWApp::onBeginGlfwMouseButtonEvent() {
    return sOnBeginGlfwMouseButtonEvent;
}

Event<GLFWwindow*, double, double>& GLFWApp::onBeginGlfwMouseCursorPosEvent() {
    return sOnBeginGlfwMouseCursorPosEvent;
}

Event<GLFWwindow*, int>& GLFWApp::onBeginGlfwMouseCursorEnterEvent() {
    return sOnBeginGlfwMouseCursorEnterEvent;
}

Event<GLFWwindow*, double, double>& GLFWApp::onBeginGlfwMouseScrollEvent() {
    return sOnBeginGlfwMouseScrollEvent;
}

Event<GLFWwindow*, unsigned int>& GLFWApp::onBeginGlfwCharEvent() {
    return sOnBeginGlfwCharEvent;
}

Event<GLFWwindow*, unsigned int, int>& GLFWApp::onBeginGlfwCharModsEvent() {
    return sOnBeginGlfwCharModsEvent;
}

Event<GLFWwindow*, int, const char**>& GLFWApp::onBeginGlfwDropEvent() {
    return sOnBeginGlfwDropEvent;
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

void GLFWApp::onKey(GLFWwindow* glfwWindow, int key, int scancode, int action,
                    int mods) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->requestRender();

    bool handled =
        sOnBeginGlfwKeyEvent(glfwWindow, key, scancode, action, mods);
    if (handled) {
        return;
    }

    window->key(key, scancode, action, mods);
}

void GLFWApp::onMouseButton(GLFWwindow* glfwWindow, int button, int action,
                            int mods) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->requestRender();

    bool handled =
        sOnBeginGlfwMouseButtonEvent(glfwWindow, button, action, mods);
    if (handled) {
        return;
    }

    window->pointerButton(button, action, mods);
}

void GLFWApp::onMouseCursorEnter(GLFWwindow* glfwWindow, int entered) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->requestRender();

    bool handled = sOnBeginGlfwMouseCursorEnterEvent(glfwWindow, entered);
    if (handled) {
        return;
    }

    window->pointerEnter(entered == GL_TRUE);
}

void GLFWApp::onMouseCursorPos(GLFWwindow* glfwWindow, double x, double y) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->requestRender();

    bool handled = sOnBeginGlfwMouseCursorPosEvent(glfwWindow, x, y);
    if (handled) {
        return;
    }

    window->pointerMoved(x, y);
}

void GLFWApp::onMouseScroll(GLFWwindow* glfwWindow, double deltaX,
                            double deltaY) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->requestRender();

    bool handled = sOnBeginGlfwMouseScrollEvent(glfwWindow, deltaX, deltaY);
    if (handled) {
        return;
    }

    window->mouseWheel(deltaX, deltaY);
}

void GLFWApp::onChar(GLFWwindow* glfwWindow, unsigned int code) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->requestRender();

    bool handled = sOnBeginGlfwCharEvent(glfwWindow, code);
    if (handled) {
        return;
    }
}

void GLFWApp::onCharMods(GLFWwindow* glfwWindow, unsigned int code, int mods) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->requestRender();

    bool handled = sOnBeginGlfwCharModsEvent(glfwWindow, code, mods);
    if (handled) {
        return;
    }
}

void GLFWApp::onDrop(GLFWwindow* glfwWindow, int numDroppedFiles,
                     const char** pathNames) {
    GLFWWindowPtr window = findWindow(glfwWindow);
    assert(window != nullptr);
    window->requestRender();

    bool handled =
        sOnBeginGlfwDropEvent(glfwWindow, numDroppedFiles, pathNames);
    if (handled) {
        return;
    }

    // TODO: Handle from Window
}

void GLFWApp::onErrorEvent(int error, const char* description) {
    JET_ERROR << "GLFW Error [" << error << "] " << description;
}

#endif  // JET_USE_GL
