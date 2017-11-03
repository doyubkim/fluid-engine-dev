// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GLFW_IMGUI_UTILS_H_
#define INCLUDE_JET_VIZ_GLFW_IMGUI_UTILS_H_

#ifdef JET_USE_GL

// Header-only utility collection for integrating ImGui with Jet.Viz.

#include <jet.viz/glfw_app.h>
#include <jet.viz/glfw_window.h>

namespace jet {

namespace viz {

// ImGui Issue #1206 (https://github.com/ocornut/imgui/issues/1206)
// Some UI features require number of frames to redisplay when
// glfwWaitEvents() is on hold.
constexpr int kImGuiRenderRequestFrames = 3;

class ImGuiForGLFWApp {
 public:
    static void configureApp() {
        GLFWApp::onBeginGlfwKeyEvent() += onBeginGlfwKey;
        GLFWApp::onBeginGlfwMouseButtonEvent() += onBeginGlfwMouseButton;
        GLFWApp::onBeginGlfwMouseScrollEvent() += onBeginGlfwMouseScroll;
        GLFWApp::onBeginGlfwCharEvent() += onBeginGlfwChar;
    }
    static void configureWindow(GLFWWindowPtr window) {
        ImGui_ImplGlfwGL3_Init(window->glfwWindow(), false);
        window->requestRender(kImGuiRenderRequestFrames);
    }

    static bool onBeginGlfwKey(GLFWwindow* glfwWindow, int key, int scancode,
                               int action, int mods) {
        ImGui_ImplGlfwGL3_KeyCallback(glfwWindow, key, scancode, action, mods);
        GLFWApp::findWindow(glfwWindow)
            ->requestRender(kImGuiRenderRequestFrames);

        return ImGui::GetIO().WantCaptureKeyboard;
    }

    static bool onBeginGlfwMouseButton(GLFWwindow* glfwWindow, int button,
                                       int action, int mods) {
        ImGui_ImplGlfwGL3_MouseButtonCallback(glfwWindow, button, action, mods);
        GLFWApp::findWindow(glfwWindow)
            ->requestRender(kImGuiRenderRequestFrames);

        return ImGui::GetIO().WantCaptureMouse;
    }

    static bool onBeginGlfwMouseScroll(GLFWwindow* glfwWindow, double deltaX,
                                       double deltaY) {
        ImGui_ImplGlfwGL3_ScrollCallback(glfwWindow, deltaX, deltaY);
        GLFWApp::findWindow(glfwWindow)
            ->requestRender(kImGuiRenderRequestFrames);

        return ImGui::GetIO().WantCaptureMouse;
    }

    static bool onBeginGlfwChar(GLFWwindow* glfwWindow, unsigned int code) {
        ImGui_ImplGlfwGL3_CharCallback(glfwWindow, code);
        GLFWApp::findWindow(glfwWindow)
            ->requestRender(kImGuiRenderRequestFrames);

        return ImGui::GetIO().WantCaptureKeyboard;
    }
};

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GLFW_APP_H_
