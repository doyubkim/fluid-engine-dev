// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "image_renderable_tests.h"
#include "points_renderable3_tests.h"

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw_gl3.h>

#include <cmath>

#include <imgui/ImGuiUtils.h>

#include <jet.viz/glfw_imgui_utils-ext.h>
#include <jet.viz/jet.viz.h>
#include <jet/jet.h>

#include <GLFW/glfw3.h>

using namespace jet;
using namespace viz;

static std::vector<OpenGLTestsPtr> sTests;
static size_t sCurrentTestIdx = 0;
static GLFWWindow* sWindow;

void nextTests() {
    sWindow->renderer()->clearRenderables();

    ++sCurrentTestIdx;
    if (sCurrentTestIdx == sTests.size()) {
        sCurrentTestIdx = 0;
    }

    sTests[sCurrentTestIdx]->setup(sWindow);
}

bool onKeyDown(GLFWWindow* win, const KeyEvent& keyEvent) {
    // "Enter" key for toggling animation
    if (keyEvent.key() == GLFW_KEY_ENTER) {
        win->setIsAnimationEnabled(!win->isAnimationEnabled());
        return true;
    } else if (keyEvent.key() == GLFW_KEY_SPACE) {
        nextTests();
        return true;
    }

    return false;
}

bool onGui(GLFWWindow*) {
    ImGui_ImplGlfwGL3_NewFrame();

    ImGui::Begin("Info");
    {
        ImGui::Text(
            "%s",
            ("Current test set #: " + std::to_string(sCurrentTestIdx)).c_str());
        if (ImGui::Button("Next test set")) {
            nextTests();
        }
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
    }
    ImGui::End();

    ImGui::Render();

    return true;
}

bool onUpdate(GLFWWindow*) { return false; }

int main(int, const char**) {
    GLFWApp::initialize();

    // Create GLFW window
    GLFWWindowPtr window = GLFWApp::createWindow("OpenGL Tests", 1280, 720);
    sWindow = window.get();

    // Setup ImGui binding
    ImGuiForGLFWApp::configureApp();
    ImGuiForGLFWApp::configureWindow(window);
    ImGui::SetupImGuiStyle(true, 0.75f);

    // Setup tests
    sTests.push_back(std::make_shared<ImageRenderableTests>(true));
    sTests.push_back(std::make_shared<ImageRenderableTests>(false));
    sTests.push_back(std::make_shared<PointsRenderable3Tests>());
    sTests[sCurrentTestIdx]->setup(window.get());

    // Set up event handlers
    window->onKeyDownEvent() += onKeyDown;
    window->onGuiEvent() += onGui;
    window->onUpdateEvent() += onUpdate;

    GLFWApp::run();

    return 0;
}