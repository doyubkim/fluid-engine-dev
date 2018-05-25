// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "example_app.h"

#ifdef JET_USE_GL
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw_gl3.h>
#endif

#include <atomic>
#include <cmath>
#include <mutex>

#ifdef JET_USE_GL
#include <imgui/ImGuiUtils.h>
#include <jet.viz/glfw_imgui_utils-ext.h>
#include <jet.viz/jet.viz.h>
#endif

#include <jet/jet.h>

#ifdef JET_USE_GL
#include <GLFW/glfw3.h>
#endif

using namespace jet;
using namespace viz;

namespace {

std::vector<ExamplePtr> sTests;
size_t sCurrentTestIdx = 0;
std::atomic<double> sSimTime{0.0};
std::atomic_bool sSimEnabled{false};
std::mutex sSimMutex;
#ifdef JET_USE_GL
GlfwWindowPtr sWindow;
#endif

#ifdef JET_USE_GL
void toggleUpdate(GlfwWindow* win, bool onoff) {
    win->setIsUpdateEnabled(onoff);
    sSimEnabled = onoff;
}

void toggleUpdate(GlfwWindow* win) { toggleUpdate(win, !sSimEnabled); }
#else
void toggleUpdate(bool onoff) { sSimEnabled = onoff; }

void toggleUpdate() { toggleUpdate(!sSimEnabled); }
#endif

#ifdef JET_USE_GL
void nextTests(GlfwWindow* win) {
    sWindow->renderer()->clearRenderables();
    toggleUpdate(win, false);
#else
void nextTests() {
    toggleUpdate(false);
#endif

    {
        std::lock_guard<std::mutex> lock(sSimMutex);
        ++sCurrentTestIdx;
        if (sCurrentTestIdx == sTests.size() || sTests.empty()) {
            sCurrentTestIdx = 0;
        }

#ifdef JET_USE_GL
        if (sTests.size() > 0) {
            sTests[sCurrentTestIdx]->setup(sWindow.get());
        }
#else
        sTests[sCurrentTestIdx]->setup();
#endif
    }
}  // namespace

#ifdef JET_USE_GL
bool onKeyDown(GlfwWindow* win, const KeyEvent& keyEvent) {
    // "Enter" key for toggling animation
    if (keyEvent.key() == GLFW_KEY_ENTER) {
        toggleUpdate(win);
        return true;
    } else if (keyEvent.key() == GLFW_KEY_SPACE) {
        nextTests(win);
        return true;
    }

    return false;
}

bool onGui(GlfwWindow* window) {
    ImGui_ImplGlfwGL3_NewFrame();

    ImGui::Begin("Info");
    {
        ImGui::Text("%s",
                    ("Testing " + sTests[sCurrentTestIdx]->name()).c_str());

        if (ImGui::Button("Next test")) {
            nextTests(window);
        }

        if (ImGui::Button("Restart sim")) {
            std::lock_guard<std::mutex> lock(sSimMutex);
            sTests[sCurrentTestIdx]->restartSim();
        }

        std::string toggleSimText("Run sim");
        if (sSimEnabled) {
            toggleSimText = "Stop sim";
        }

        if (ImGui::Button(toggleSimText.c_str())) {
            toggleUpdate(window);
        }

        ImGui::Text("Rendering average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);

        const double simTime = sSimTime;
        ImGui::Text("Simulation average %.3f ms/frame (%.1f FPS)",
                    simTime * 1000.0, 1.0 / simTime);
    }
    ImGui::End();

    if (sTests.size() > 0) {
        sTests[sCurrentTestIdx]->gui(window);
    }

    ImGui::Render();

    return true;
}
#endif

}  // namespace

#ifdef JET_USE_GL
bool onUpdateRenderables(GlfwWindow*) {
    if (sTests.size() > 0) {
        sTests[sCurrentTestIdx]->updateRenderables();
    }
    return true;
}
#endif

void ExampleApp::initialize(const std::string& appName, int windowWidth,
                            int windowHeight) {
#ifdef JET_USE_GL
    GlfwApp::initialize();

    // Create GLFW window
    sWindow = GlfwApp::createWindow(appName, windowWidth, windowHeight);

    // Setup ImGui binding
    ImGuiForGlfwApp::configureApp();
    ImGuiForGlfwApp::configureWindow(sWindow.get());
    ImGui::SetupImGuiStyle(true, 0.75f);

    // Set up event handlers
    sWindow->onKeyDownEvent() += onKeyDown;
    sWindow->onGuiEvent() += onGui;
    sWindow->onUpdateEvent() += onUpdateRenderables;
#else
    (void)windowWidth;
    (void)windowHeight;
    printf("Starting example \"%s\"\n", appName.c_str());
#endif
}

void ExampleApp::finalize() {
#ifdef JET_USE_GL
    // Clears the device memory before main() terminates
    sWindow->renderer()->clearRenderables();
#endif
    sTests.clear();
}

void ExampleApp::addExample(const ExamplePtr& example) {
    sTests.push_back(example);
}

void ExampleApp::run() {
#ifdef JET_USE_GL
    sTests[sCurrentTestIdx]->setup(sWindow.get());
#else
    sTests[sCurrentTestIdx]->setup();
#endif

    // Worker thread for sim
    bool done = false;
    std::thread worker([&]() {
        Timer timer;

        while (!done) {
            if (sSimEnabled) {
                std::lock_guard<std::mutex> lock(sSimMutex);
                timer.reset();
                if (sTests.size() > 0) {
                    sTests[sCurrentTestIdx]->advanceSim();
                }
                sSimTime = timer.durationInSeconds();
            }
        }
    });

#ifdef JET_USE_GL
    GlfwApp::run();
#endif
    done = true;
    worker.join();
}
