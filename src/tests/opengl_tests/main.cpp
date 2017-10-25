// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet.viz/jet.viz.h>
#include <jet/jet.h>

#define kEnterKey 257

using namespace jet;
using namespace viz;

int main(int, const char**) {
    GLFWApp::initialize();

    // Create GLFW window
    GLFWWindowPtr window = GLFWApp::createWindow("OpenGL Test", 512, 512);
    window->setViewController(
        std::make_shared<OrthoViewController>(std::make_shared<OrthoCamera>()));

    //
    auto renderer = window->renderer();

    auto renderable = std::make_shared<PointsRenderable2>(renderer.get());
    std::vector<Vector2F> positions = {
        {-0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f}, {0.5f, -0.5f}};
    std::vector<Color> colors(4, Color{1.0f, 1.0f, 1.0f, 1.0f});
    renderable->setPositionsAndColors(positions.data(), colors.data(),
                                      positions.size());
    renderable->setRadius(5.0f);
    renderer->addRenderable(renderable);

    // Set up event handlers
    window->onKeyDownEvent() += [](GLFWWindow* win, const KeyEvent& keyEvent) {
        // "Enter" key for toggling animation
        if (keyEvent.key() == kEnterKey) {
            win->setIsAnimationEnabled(!win->isAnimationEnabled());
        }
    };

    Frame frame{0, 1.0 / 60.0};
    window->onUpdateEvent() += [&frame](GLFWWindow*) { ++frame; };

    window->requestRender();

    GLFWApp::run();

    return 0;
}