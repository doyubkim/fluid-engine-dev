// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet.viz/jet.viz.h>
#include <jet/jet.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw_gl3.h>

#include <imgui/ImGuiUtils.h>

#include <GLFW/glfw3.h>

using namespace jet;
using namespace viz;

// ImGui Issue #1206 (https://github.com/ocornut/imgui/issues/1206)
// Some UI features require number of frames to redisplay when
// glfwWaitEvents() is on hold.
#define IMGUI_RENDER_REQUEST_FRAMES 3

bool onKeyDown(GLFWWindow* win, const KeyEvent& keyEvent) {
    // "Enter" key for toggling animation
    if (keyEvent.key() == GLFW_KEY_ENTER) {
        win->setIsAnimationEnabled(!win->isAnimationEnabled());
        return true;
    }

    return false;
}

bool onGui(GLFWWindow*) {
    static float f = 0.0f;
    static ImVec4 clear_color = ImColor(114, 144, 154);

    ImGui_ImplGlfwGL3_NewFrame();

    ImGui::Text("Hello, world!");
    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
    ImGui::ColorEdit3("clear color", (float*)&clear_color);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Render();

    return true;
}

bool onUpdate(GLFWWindow*) { return false; }

bool onBeginGlfwKey(GLFWwindow* glfwWindow, int key, int scancode, int action,
                    int mods) {
    ImGui_ImplGlfwGL3_KeyCallback(glfwWindow, key, scancode, action, mods);
    GLFWApp::findWindow(glfwWindow)->requestRender(IMGUI_RENDER_REQUEST_FRAMES);

    return ImGui::GetIO().WantCaptureKeyboard;
}

bool onBeginGlfwMouseButton(GLFWwindow* glfwWindow, int button, int action,
                            int mods) {
    ImGui_ImplGlfwGL3_MouseButtonCallback(glfwWindow, button, action, mods);
    GLFWApp::findWindow(glfwWindow)->requestRender(IMGUI_RENDER_REQUEST_FRAMES);

    return ImGui::GetIO().WantCaptureMouse;
}

bool onBeginGlfwMouseScroll(GLFWwindow* glfwWindow, double deltaX,
                            double deltaY) {
    ImGui_ImplGlfwGL3_ScrollCallback(glfwWindow, deltaX, deltaY);
    GLFWApp::findWindow(glfwWindow)->requestRender(IMGUI_RENDER_REQUEST_FRAMES);

    return ImGui::GetIO().WantCaptureMouse;
}

bool onBeginGlfwChar(GLFWwindow* glfwWindow, unsigned int code) {
    ImGui_ImplGlfwGL3_CharCallback(glfwWindow, code);
    GLFWApp::findWindow(glfwWindow)->requestRender(IMGUI_RENDER_REQUEST_FRAMES);

    return ImGui::GetIO().WantCaptureKeyboard;
}

int main(int, const char**) {
    GLFWApp::initialize();

    // Create GLFW window
    GLFWWindowPtr window = GLFWApp::createWindow("OpenGL Test", 1280, 720);

    // Setup ImGui binding
    ImGui_ImplGlfwGL3_Init(window->glfwWindow(), false);
    GLFWApp::onBeginGlfwKeyEvent() += onBeginGlfwKey;
    GLFWApp::onBeginGlfwMouseButtonEvent() += onBeginGlfwMouseButton;
    GLFWApp::onBeginGlfwMouseScrollEvent() += onBeginGlfwMouseScroll;
    GLFWApp::onBeginGlfwCharEvent() += onBeginGlfwChar;
    ImGui::SetupImGuiStyle(true, 0.75f);
    window->requestRender(IMGUI_RENDER_REQUEST_FRAMES);

    window->setViewController(
        std::make_shared<OrthoViewController>(std::make_shared<OrthoCamera>()));

    // Setup renderer
    auto renderer = window->renderer();
    renderer->setBackgroundColor(Color{1, 1, 1, 1});
    const ByteImage img(128, 128, ByteColor::makeGreen());
    const auto renderable = std::make_shared<ImageRenderable>(renderer.get());
    renderable->setImage(img);
    renderer->addRenderable(renderable);

    // Set up event handlers
    window->onKeyDownEvent() += onKeyDown;
    window->onGuiEvent() += onGui;
    window->onUpdateEvent() += onUpdate;

    GLFWApp::run();

    return 0;
}