// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_particle_system_solver3_example.h"
#include "cuda_pci_sph_solver3_example.h"
#include "cuda_wc_sph_solver3_example.h"
#include "pci_sph_solver3_example.h"

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

static std::vector<ParticleSimExamplePtr> sTests;
static size_t sCurrentTestIdx = 0;
static GlfwWindow* sWindow;

void nextTests() {
    sWindow->renderer()->clearRenderables();

    ++sCurrentTestIdx;
    if (sCurrentTestIdx == sTests.size()) {
        sCurrentTestIdx = 0;
    }

    sTests[sCurrentTestIdx]->setup(sWindow);
}

bool onKeyDown(GlfwWindow* win, const KeyEvent& keyEvent) {
    // "Enter" key for toggling animation
    if (keyEvent.key() == GLFW_KEY_ENTER) {
        win->setIsUpdateEnabled(!win->isUpdateEnabled());
        return true;
    } else if (keyEvent.key() == GLFW_KEY_SPACE) {
        nextTests();
        return true;
    }

    return false;
}

bool onGui(GlfwWindow* window) {
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

    sTests[sCurrentTestIdx]->gui(window);

    ImGui::Render();

    return true;
}

bool onUpdate(GlfwWindow*) {
    sTests[sCurrentTestIdx]->update();
    return true;
}

int main(int, const char**) {
    Logging::mute();

    GlfwApp::initialize();

    // Create GLFW window
    GlfwWindowPtr window =
        GlfwApp::createWindow("CUDA Particle Sim", 1280, 720);
    sWindow = window.get();

    // Setup ImGui binding
    ImGuiForGlfwApp::configureApp();
    ImGuiForGlfwApp::configureWindow(window);
    ImGui::SetupImGuiStyle(true, 0.75f);

    // Setup tests
    sTests.push_back(
        std::make_shared<CudaPciSphSolver3Example>(Frame(0, 1.0 / 3000.0)));
    sTests.push_back(
        std::make_shared<PciSphSolver3Example>(Frame(0, 1.0 / 3000.0)));
    sTests.push_back(
        std::make_shared<CudaWcSphSolver3Example>(Frame(0, 1.0 / 3000.0)));
    sTests.push_back(std::make_shared<CudaParticleSystemSolver3Example>(
        Frame(0, 1.0 / 1000.0)));
    sTests[sCurrentTestIdx]->setup(window.get());

    // Set up event handlers
    window->onKeyDownEvent() += onKeyDown;
    window->onGuiEvent() += onGui;
    window->onUpdateEvent() += onUpdate;

    GlfwApp::run();

    // Clears the device memory before main() terminates
    sWindow->renderer()->clearRenderables();
    sTests.clear();

    return 0;
}
