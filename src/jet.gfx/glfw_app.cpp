// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_GL

#include <common.h>

#include <jet.gfx/gl_common.h>
#include <jet.gfx/glfw_app.h>
#include <jet.gfx/glfw_window.h>

namespace jet {
namespace gfx {

namespace {

std::vector<GlfwWindowPtr> sWindows;
GlfwWindowPtr sCurrentWindow;

Event<GLFWwindow *, int, int> sOnGlfwWindowResizedEvent;
Event<GLFWwindow *, int, int, int, int> sOnGlfwKeyEvent;
Event<GLFWwindow *, int, int, int> sOnGlfwMouseButtonEvent;
Event<GLFWwindow *, double, double> sOnGlfwMouseCursorPosEvent;
Event<GLFWwindow *, int> sOnGlfwMouseCursorEnterEvent;
Event<GLFWwindow *, double, double> sOnGlfwMouseScrollEvent;
Event<GLFWwindow *, unsigned int> sOnGlfwCharEvent;
Event<GLFWwindow *, unsigned int, int> sOnGlfwCharModsEvent;
Event<GLFWwindow *, int, const char **> sOnGlfwDropEvent;

}  // namespace

int GlfwApp::initialize() {
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

int GlfwApp::run() {
    // Force render first frame
    if (sCurrentWindow != nullptr) {
        sCurrentWindow->requestRender(1);
    }

    while (sCurrentWindow != nullptr) {
        glfwWaitEvents();

        auto window = sCurrentWindow->glfwWindow();

        if (sCurrentWindow->isUpdateEnabled() ||
            sCurrentWindow->_numRequestedRenderFrames > 0) {
            if (sCurrentWindow->isUpdateEnabled()) {
                sCurrentWindow->update();
            }

            sCurrentWindow->render();

            // Decrease render request count
            sCurrentWindow->_numRequestedRenderFrames -= 1;

            if (sCurrentWindow->isUpdateEnabled()) {
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

GlfwWindowPtr GlfwApp::createWindow(const std::string &title, int width,
                                    int height) {
    sCurrentWindow = GlfwWindowPtr(new GlfwWindow(title, width, height));
    sWindows.push_back(sCurrentWindow);

    auto glfwWindow = sCurrentWindow->glfwWindow();

    glfwSetWindowSizeCallback(glfwWindow, onWindowResized);
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

GlfwWindowPtr GlfwApp::findWindow(GLFWwindow *glfwWindow) {
    for (auto w : sWindows) {
        if (w->glfwWindow() == glfwWindow) {
            return w;
        }
    }

    return nullptr;
}

Event<GLFWwindow *, int, int, int, int> &GlfwApp::onGlfwKeyEvent() {
    return sOnGlfwKeyEvent;
}

Event<GLFWwindow *, int, int, int> &GlfwApp::onGlfwMouseButtonEvent() {
    return sOnGlfwMouseButtonEvent;
}

Event<GLFWwindow *, double, double> &GlfwApp::onGlfwMouseCursorPosEvent() {
    return sOnGlfwMouseCursorPosEvent;
}

Event<GLFWwindow *, int> &GlfwApp::onGlfwMouseCursorEnterEvent() {
    return sOnGlfwMouseCursorEnterEvent;
}

Event<GLFWwindow *, double, double> &GlfwApp::onGlfwMouseScrollEvent() {
    return sOnGlfwMouseScrollEvent;
}

Event<GLFWwindow *, unsigned int> &GlfwApp::onGlfwCharEvent() {
    return sOnGlfwCharEvent;
}

Event<GLFWwindow *, unsigned int, int> &GlfwApp::onGlfwCharModsEvent() {
    return sOnGlfwCharModsEvent;
}

Event<GLFWwindow *, int, const char **> &GlfwApp::onGlfwDropEvent() {
    return sOnGlfwDropEvent;
}

void GlfwApp::onSetCurrentWindow(const GlfwWindowPtr &window) {
    JET_ASSERT(std::find(sWindows.begin(), sWindows.end(), window) !=
               sWindows.end());

    sCurrentWindow = window;
}

void GlfwApp::onCloseCurrentWindow(const GlfwWindowPtr &window) {
    auto it = std::find(sWindows.begin(), sWindows.end(), window);
    sWindows.erase(it);

    if (sCurrentWindow == window) {
        sCurrentWindow.reset();

        if (!sWindows.empty()) {
            sCurrentWindow = *sWindows.rbegin();
        }
    }
}

void GlfwApp::onWindowResized(GLFWwindow *glfwWindow, int width, int height) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->resize(width, height);
    window->requestRender(1);

    JET_DEBUG << "GLFW Window resized to " << width << " x " << height;

    sOnGlfwWindowResizedEvent(glfwWindow, width, height);
}

void GlfwApp::onKey(GLFWwindow *glfwWindow, int key, int scancode, int action,
                    int mods) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->requestRender(1);

    bool handled = sOnGlfwKeyEvent(glfwWindow, key, scancode, action, mods);
    if (handled) {
        return;
    }

    window->key(key, scancode, action, mods);
}

void GlfwApp::onMouseButton(GLFWwindow *glfwWindow, int button, int action,
                            int mods) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->requestRender(1);

    bool handled = sOnGlfwMouseButtonEvent(glfwWindow, button, action, mods);
    if (handled) {
        return;
    }

    window->pointerButton(button, action, mods);
}

void GlfwApp::onMouseCursorEnter(GLFWwindow *glfwWindow, int entered) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->requestRender(1);

    bool handled = sOnGlfwMouseCursorEnterEvent(glfwWindow, entered);
    if (handled) {
        return;
    }

    window->pointerEnter(entered == GL_TRUE);
}

void GlfwApp::onMouseCursorPos(GLFWwindow *glfwWindow, double x, double y) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->requestRender(1);

    bool handled = sOnGlfwMouseCursorPosEvent(glfwWindow, x, y);
    if (handled) {
        return;
    }

    window->pointerMoved(x, y);
}

void GlfwApp::onMouseScroll(GLFWwindow *glfwWindow, double deltaX,
                            double deltaY) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->requestRender(1);

    bool handled = sOnGlfwMouseScrollEvent(glfwWindow, deltaX, deltaY);
    if (handled) {
        return;
    }

    window->mouseWheel(deltaX, deltaY);
}

void GlfwApp::onChar(GLFWwindow *glfwWindow, unsigned int code) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->requestRender(1);

    bool handled = sOnGlfwCharEvent(glfwWindow, code);
    if (handled) {
        return;
    }
}

void GlfwApp::onCharMods(GLFWwindow *glfwWindow, unsigned int code, int mods) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->requestRender(1);

    bool handled = sOnGlfwCharModsEvent(glfwWindow, code, mods);
    if (handled) {
        return;
    }
}

void GlfwApp::onDrop(GLFWwindow *glfwWindow, int numDroppedFiles,
                     const char **pathNames) {
    GlfwWindowPtr window = findWindow(glfwWindow);
    JET_ASSERT(window != nullptr);
    window->requestRender(1);

    bool handled = sOnGlfwDropEvent(glfwWindow, numDroppedFiles, pathNames);
    if (handled) {
        return;
    }

    // TODO: Handle from Window
}

void GlfwApp::onErrorEvent(int error, const char *description) {
    JET_ERROR << "GLFW Error [" << error << "] " << description;
}

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL
