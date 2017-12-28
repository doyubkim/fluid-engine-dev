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

#include <random>

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
    sRenderable->setImage(sImage);
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
    sSolver->setEmitter(emitter);
    emitter->addStepFunctionTarget(sSolver->smokeDensity(), 0.0, 1.0);
    emitter->addStepFunctionTarget(sSolver->temperature(), 0.0,
                                   1.0 * 0.5 * sRandomDist(sRandomGen));
}

// MARK: Event handlers
bool onKeyDown(GLFWWindow* win, const KeyEvent& keyEvent) {
    switch (keyEvent.key()) {
        case 'q':
        case 'Q':
            exit(EXIT_SUCCESS);
        case GLFW_KEY_ENTER:
            win->setIsAnimationEnabled(!win->isAnimationEnabled());
            return true;
        default:
            break;
    }

    return false;
}

bool onPointerPressed(GLFWWindow* win, const PointerEvent& pointerEvent) {
    (void)win;
    (void)pointerEvent;
    return true;
}

bool onPointerReleased(GLFWWindow* win, const PointerEvent& pointerEvent) {
    (void)win;
    (void)pointerEvent;
    return true;
}

bool onPointerDragged(GLFWWindow* win, const PointerEvent& pointerEvent) {
    (void)win;
    (void)pointerEvent;
    return true;
}

bool onGui(GLFWWindow*) {
    ImGui_ImplGlfwGL3_NewFrame();
    ImGui::Begin("Info");
    {
        ImGui::Text("Application average %.3f ms/sFrame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);

        if (ImGui::Button("Add Source")) {
            resetEmitter();
        }

        auto diff = (float)sSolver->smokeDiffusionCoefficient();
        ImGui::SliderFloat("Smoke Diffusion", &diff, 0.0f, 0.005f, "%.5f");
        sSolver->setSmokeDiffusionCoefficient(diff);

        auto decay = (float)sSolver->smokeDecayFactor();
        ImGui::SliderFloat("Smoke Decay", &decay, 0.0f, 0.01f);
        sSolver->setSmokeDecayFactor(decay);
    }
    ImGui::End();
    ImGui::Render();
    return true;
}

bool onUpdate(GLFWWindow* win) {
    (void)win;

    sSolver->update(sFrame);
    densityToImage();

    ++sFrame;

    return true;
}

int main(int, const char**) {
    Logging::mute();

    GLFWApp::initialize();

    // Setup sSolver
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
    resetEmitter();

    // Create GLFW window
    GLFWWindowPtr window = GLFWApp::createWindow("Smoke Sim 2D", 512, 512);

    // Setup ImGui binding
    ImGuiForGLFWApp::configureApp();
    ImGuiForGLFWApp::configureWindow(window);
    ImGui::SetupImGuiStyle(true, 0.75f);
    window->setIsAnimationEnabled(true);

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

    GLFWApp::run();

    return 0;
}
