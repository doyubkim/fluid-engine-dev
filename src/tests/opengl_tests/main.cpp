// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet.viz/jet.viz.h>
#include <jet/jet.h>

#include <imgui/imgui.h>

#define kEnterKey 257

using namespace jet;
using namespace viz;

void onKeyDown(GLFWWindow* win, const KeyEvent& keyEvent) {
    // "Enter" key for toggling animation
    if (keyEvent.key() == kEnterKey) {
        win->setIsAnimationEnabled(!win->isAnimationEnabled());
    }
}

void onGui(GLFWWindow*) {
#ifdef JET_USE_IMGUI
    static float f = 0.0f;
    static ImVec4 clear_color = ImColor(114, 144, 154);
    ImGui::Text("Hello, world!");
    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
    ImGui::ColorEdit3("clear color", (float*)&clear_color);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
#endif  // JET_USE_IMGUI
}

void onUpdate(GLFWWindow*) {}

int main(int, const char**) {
    GLFWApp::initialize();

    // Create GLFW window
    GLFWWindowPtr window = GLFWApp::createWindow("OpenGL Test", 512, 512);

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