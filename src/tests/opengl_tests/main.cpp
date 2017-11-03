// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw_gl3.h>

#include <cmath>
#include <imgui/ImGuiUtils.h>

#include <jet.viz/jet.viz.h>
#include <jet/jet.h>

#include <GLFW/glfw3.h>

using namespace jet;
using namespace viz;

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

int main(int, const char**) {
    GLFWApp::initialize();

    // Create GLFW window
    GLFWWindowPtr window = GLFWApp::createWindow("OpenGL Test", 1280, 720);

    // Setup ImGui binding
    ImGuiForGLFWApp::configureApp();
    ImGuiForGLFWApp::configureWindow(window);
    ImGui::SetupImGuiStyle(true, 0.75f);

    window->setViewController(
        std::make_shared<OrthoViewController>(std::make_shared<OrthoCamera>()));

    // Setup renderer
    auto renderer = window->renderer();
    renderer->setBackgroundColor(Color{1, 1, 1, 1});

    // Load sample image renderable
    const ByteImage img(RESOURCES_DIR "/airplane.png");
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