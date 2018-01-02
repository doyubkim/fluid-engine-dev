// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

// MARK: Global variables
static size_t sN = 128;
static Frame sFrame{0, 1.0 / 60.0};
static GridSmokeSolver2Ptr sSolver;
static ImageRenderablePtr sRenderable;
static ByteImage sImage;
static VolumeGridEmitter2Ptr sEmitter;

std::mt19937 sRandomGen(0);
std::uniform_real_distribution<> sRandomDist(0.0, 1.0);

std::atomic_bool sIsImageDirty{true};
std::atomic<double> sSimTime{0.0};
std::atomic<double> sSimDiffCoeff{0.0};
std::atomic<double> sSimDecayCoeff{0.0};
std::mutex sSimMutex;

// MARK: Rendering
void densityToImage() {
    const auto den = sSolver->smokeDensity()->dataAccessor();
    sImage.resize(sN, sN);
    for (size_t i = 0; i < sN; i++) {
        for (size_t j = 0; j < sN; j++) {
            const float d = (float)clamp(2.0 * den(i, j) - 1.0, -1.0, 1.0);
            auto color = ByteColor(Color::makeJet(d));
            sImage(i, j) = color;
        }
    }
}

// MARK: Emitter
void resetEmitter() {
    auto sphere = Sphere2::builder()
                      .withCenter({0.2 + 0.6 * sRandomDist(sRandomGen),
                                   0.2 + 0.3 * sRandomDist(sRandomGen)})
                      .withRadius(0.05 + 0.1 * sRandomDist(sRandomGen))
                      .makeShared();
    auto emitter =
        VolumeGridEmitter2::builder().withSourceRegion(sphere).makeShared();

    std::lock_guard<std::mutex> lock(sSimMutex);
    sSolver->setEmitter(emitter);
    emitter->addStepFunctionTarget(sSolver->smokeDensity(), 0.0, 1.0);
    emitter->addStepFunctionTarget(sSolver->temperature(), 0.0,
                                   1.0 * 0.5 * sRandomDist(sRandomGen));
}

// MARK: Event handlers
bool onKeyDown(GlfwWindow* win, const KeyEvent& keyEvent) {
    switch (keyEvent.key()) {
        case 'q':
        case 'Q':
            exit(EXIT_SUCCESS);
        case GLFW_KEY_ENTER:
            win->setIsUpdateEnabled(!win->isUpdateEnabled());
            return true;
        default:
            break;
    }

    return false;
}

bool onPointerPressed(GlfwWindow* win, const PointerEvent& pointerEvent) {
    (void)win;
    (void)pointerEvent;
    return true;
}

bool onPointerReleased(GlfwWindow* win, const PointerEvent& pointerEvent) {
    (void)win;
    (void)pointerEvent;
    return true;
}

bool onPointerDragged(GlfwWindow* win, const PointerEvent& pointerEvent) {
    (void)win;
    (void)pointerEvent;
    return true;
}

bool onGui(GlfwWindow*) {
    ImGui_ImplGlfwGL3_NewFrame();
    ImGui::Begin("Info");
    {
        ImGui::Text("Rendering average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);

        const double simTime = sSimTime;
        ImGui::Text("Simulation average %.3f ms/frame (%.1f FPS)",
                    simTime * 1000.0, 1.0 / simTime);

        if (ImGui::Button("Add Source")) {
            resetEmitter();
        }

        auto diff = (float)sSimDiffCoeff;
        ImGui::SliderFloat("Smoke Diffusion", &diff, 0.0f, 0.005f, "%.5f");
        sSimDiffCoeff = diff;

        auto decay = (float)sSimDecayCoeff;
        ImGui::SliderFloat("Smoke Decay", &decay, 0.0f, 0.01f);
        sSimDecayCoeff = decay;
    }
    ImGui::End();
    ImGui::Render();
    return true;
}

bool onUpdate(GlfwWindow* win) {
    (void)win;

    if (sIsImageDirty) {
        sRenderable->setImage(sImage);
        sIsImageDirty = false;
    }

    return true;
}

int main(int, const char**) {
    Logging::mute();

    GlfwApp::initialize();

    // Setup solver
    sSolver = GridSmokeSolver2::builder()
                  .withResolution({sN, sN})
                  .withDomainSizeX(1.0)
                  .makeShared();
    auto pressureSolver =
        std::make_shared<GridFractionalSinglePhasePressureSolver2>();
    pressureSolver->setLinearSystemSolver(
        std::make_shared<FdmMgSolver2>(6, 3, 3, 3, 3));
    auto diffusionSolver =
        std::make_shared<GridBackwardEulerDiffusionSolver2>();
    diffusionSolver->setLinearSystemSolver(
        std::make_shared<FdmGaussSeidelSolver2>(10, 10, 1e-3));
    sSolver->setPressureSolver(pressureSolver);
    sSolver->setDiffusionSolver(diffusionSolver);
    sSimDiffCoeff = sSolver->smokeDiffusionCoefficient();
    sSimDecayCoeff = sSolver->smokeDecayFactor();
    resetEmitter();

    // Create GLFW window
    GlfwWindowPtr window = GlfwApp::createWindow("Smoke Sim 2D", 512, 512);

    // Setup ImGui binding
    ImGuiForGlfwApp::configureApp();
    ImGuiForGlfwApp::configureWindow(window);
    ImGui::SetupImGuiStyle(true, 0.75f);
    window->setIsUpdateEnabled(true);

    auto camera = std::make_shared<OrthoCamera>(-0.5, 0.5, -0.5, 0.5);
    auto viewController = std::make_shared<OrthoViewController>(camera);
    viewController->enablePan = false;
    viewController->enableZoom = false;
    viewController->enableRotation = false;
    window->setViewController(viewController);

    // Setup renderer
    auto renderer = window->renderer();
    renderer->setBackgroundColor(Color{1, 1, 1, 1});

    // Load sample image renderable
    const ByteImage img(sN, sN, ByteColor::makeBlack());
    sRenderable = std::make_shared<ImageRenderable>(renderer.get());
    sRenderable->setImage(img);
    sRenderable->setTextureSamplingMode(TextureSamplingMode::kLinear);
    renderer->addRenderable(sRenderable);

    // Set up event handlers
    window->onKeyDownEvent() += onKeyDown;
    window->onPointerPressedEvent() += onPointerPressed;
    window->onPointerReleasedEvent() += onPointerReleased;
    window->onPointerDraggedEvent() += onPointerDragged;
    window->onGuiEvent() += onGui;
    window->onUpdateEvent() += onUpdate;

    // Worker thread for sim
    bool done = false;
    std::thread t([&]() {
        Timer timer;

        while (!done) {
            std::lock_guard<std::mutex> lock(sSimMutex);

            timer.reset();
            sSolver->setSmokeDiffusionCoefficient(sSimDiffCoeff);
            sSolver->setSmokeDecayFactor(sSimDecayCoeff);
            sSolver->update(sFrame);
            sSimTime = timer.durationInSeconds();

            ++sFrame;
            densityToImage();
            sIsImageDirty = true;
        }
    });

    GlfwApp::run();
    done = true;
    t.join();

    return 0;
}
