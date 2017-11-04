// Original code by Jos Stam, modified by Doyub Kim
//
// Author : Jos Stam (jstam@aw.sgi.com)
// Creation Date : Jan 9 2003
//
// Description:
//
// This code is a simple prototype that demonstrates how to use the
// code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
// for Games". This code uses OpenGL and GLUT for graphics and interface
//
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

// MARK: Macros
#define IX(i, j) ((i) + (N + 2) * (j))

// MARK: External definitions (from solver.cpp)
extern void dens_step(int N, float* x, float* x0, float* u, float* v,
                      float diff, float dt);
extern void vel_step(int N, float* u, float* v, float* u0, float* v0,
                     float visc, float dt);

// MARK: Global variables
static int N;
static float dt, diff, visc;
static float force, source;
static int dvel;

static float *u, *v, *uPrev, *vPrev;
static float *dens, *densPrev;

static int frameX, frameY;
static int mouseDown[2];
static int omx, omy, mx, my;

static ImageRenderablePtr sRenderable;
static ByteImage sImage;

// MARK: Free/clear/allocate simulation data
static void freeData(void) {
    if (u) free(u);
    if (v) free(v);
    if (uPrev) free(uPrev);
    if (vPrev) free(vPrev);
    if (dens) free(dens);
    if (densPrev) free(densPrev);
}

static void clearData(void) {
    int i, size = (N + 2) * (N + 2);

    for (i = 0; i < size; i++) {
        u[i] = v[i] = uPrev[i] = vPrev[i] = dens[i] = densPrev[i] = 0.0f;
    }
}

static bool allocateData(void) {
    int size = (N + 2) * (N + 2);

    u = (float*)malloc(size * sizeof(float));
    v = (float*)malloc(size * sizeof(float));
    uPrev = (float*)malloc(size * sizeof(float));
    vPrev = (float*)malloc(size * sizeof(float));
    dens = (float*)malloc(size * sizeof(float));
    densPrev = (float*)malloc(size * sizeof(float));

    if (!u || !v || !uPrev || !vPrev || !dens || !densPrev) {
        fprintf(stderr, "cannot allocate data\n");
        return false;
    }

    return true;
}

// MARK: Relates mouse movements to forces sources
static void getFromUI(float* d, float* u, float* v) {
    int i, j, size = (N + 2) * (N + 2);

    for (i = 0; i < size; i++) {
        u[i] = v[i] = d[i] = 0.0f;
    }

    if (!mouseDown[0] && !mouseDown[1]) return;

    i = (int)((mx / (float)frameX) * N + 1);
    j = (int)(((frameY - my) / (float)frameY) * N + 1);

    if (i < 1 || i > N || j < 1 || j > N) return;

    if (mouseDown[0]) {
        u[IX(i, j)] = force * (mx - omx);
        v[IX(i, j)] = force * (omy - my);
    }

    if (mouseDown[1]) {
        d[IX(i, j)] = source;
    }

    omx = mx;
    omy = my;
}

// MARK: Rendering
void densityToImage() {
    sImage.resize(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            const float d = clamp(dens[IX(i, j)], 0.0f, 1.0f);
            const auto bd = static_cast<uint8_t>(d * 255);
            ByteColor color{bd, bd, bd, 255};
            sImage(i, j) = color;
        }
    }
    sRenderable->setImage(sImage);
}

// MARK: Event handlers
bool onKeyDown(GLFWWindow* win, const KeyEvent& keyEvent) {
    switch (keyEvent.key()) {
        case 'c':
        case 'C':
            clearData();
            return true;

        case 'q':
        case 'Q':
            freeData();
            exit(0);

        case 'v':
        case 'V':
            dvel = !dvel;
            return true;

        case GLFW_KEY_ENTER:
            win->setIsAnimationEnabled(!win->isAnimationEnabled());
            return true;
    }

    return false;
}

bool onPointerPressed(GLFWWindow* win, const PointerEvent& pointerEvent) {
    int button =
        (pointerEvent.pressedMouseButton() == MouseButtonType::Left) ? 0 : 1;
    mouseDown[button] = true;
    return true;
}

bool onPointerReleased(GLFWWindow* win, const PointerEvent& pointerEvent) {
    int button =
        (pointerEvent.pressedMouseButton() == MouseButtonType::Left) ? 0 : 1;
    mouseDown[button] = false;
    return true;
}

bool onPointerDragged(GLFWWindow* win, const PointerEvent& pointerEvent) {
    mx = (int)pointerEvent.x();
    my = (int)pointerEvent.y();
    return true;
}

bool onGui(GLFWWindow*) {
    ImGui_ImplGlfwGL3_NewFrame();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Render();
    return true;
}

bool onUpdate(GLFWWindow* win) {
    const auto fbSize = win->framebufferSize();
    frameX = (int)fbSize.x;
    frameY = (int)fbSize.y;

    getFromUI(densPrev, uPrev, vPrev);
    vel_step(N, u, v, uPrev, vPrev, visc, dt);
    dens_step(N, dens, densPrev, u, v, diff, dt);

    densityToImage();

    return true;
}

int main(int argc, const char** argv) {
    GLFWApp::initialize();

    // Setup solver
    if (argc != 1 && argc != 6) {
        fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
        fprintf(stderr, "where:\n");
        fprintf(stderr, "\t N      : grid resolution\n");
        fprintf(stderr, "\t dt     : time step\n");
        fprintf(stderr, "\t diff   : diffusion rate of the density\n");
        fprintf(stderr, "\t visc   : viscosity of the fluid\n");
        fprintf(
            stderr,
            "\t force  : scales the mouse movement that generate a force\n");
        fprintf(stderr,
                "\t source : amount of density that will be deposited\n");
        exit(1);
    }

    if (argc == 1) {
        N = 128;
        dt = 0.1f;
        diff = 0.0f;
        visc = 0.0f;
        force = 5.0f;
        source = 100.0f;
        fprintf(stderr,
                "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g "
                "source=%g\n",
                N, dt, diff, visc, force, source);
    } else {
        N = atoi(argv[1]);
        dt = (float)atof(argv[2]);
        diff = (float)atof(argv[3]);
        visc = (float)atof(argv[4]);
        force = (float)atof(argv[5]);
        source = (float)atof(argv[6]);
    }

    printf("\n\nHow to use this demo:\n\n");
    printf("\t Add densities with the right mouse button\n");
    printf(
        "\t Add velocities with the left mouse button and dragging the "
        "mouse\n");
    printf("\t Toggle density/velocity display with the 'v' key\n");
    printf("\t Clear the simulation by pressing the 'c' key\n");
    printf("\t Quit by pressing the 'q' key\n");

    dvel = 0;

    if (!allocateData()) exit(1);
    clearData();

    // Create GLFW window
    GLFWWindowPtr window =
        GLFWApp::createWindow("Jos Stam's Stable Fluids", 512, 512);

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
    const ByteImage img(N, N, ByteColor::makeBlack());
    sRenderable = std::make_shared<ImageRenderable>(renderer.get());
    sRenderable->setImage(img);
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