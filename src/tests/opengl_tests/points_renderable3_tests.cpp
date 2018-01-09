// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "points_renderable3_tests.h"

#include <random>

using namespace jet;
using namespace viz;

void PointsRenderable3Tests::setup(GlfwWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>()));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{1, 1, 1, 1});

    // Generate random points
    std::vector<Vector3F> positions;
    std::vector<Color> colors;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> d1(-0.5f, 0.5f);
    std::uniform_real_distribution<float> d2(-1.0f, 1.0f);
    for (size_t i = 0; i < 1000; ++i) {
        positions.emplace_back(d1(rng), d1(rng), d1(rng));
        colors.push_back(Color::makeJet(d2(rng)));
    }

    auto renderable = std::make_shared<PointsRenderable3>(renderer);
    renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                          window->windowSize().x);
    renderable->setPositionsAndColors(positions.data(), colors.data(),
                                      positions.size());

    renderer->addRenderable(renderable);
}
