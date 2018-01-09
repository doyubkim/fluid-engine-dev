// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_vbo_tests.h"

#include <jet.viz/points_renderable3.h>

#include <thrust/device_vector.h>

#include <random>

using namespace jet;
using namespace viz;

void CudaVboTests::setup(GlfwWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>()));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{1, 1, 1, 1});

    // Generate random points
    thrust::device_vector<VertexPosition3Color4> vertices;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> d1(-0.5f, 0.5f);
    std::uniform_real_distribution<float> d2(-1.0f, 1.0f);
    for (size_t i = 0; i < 1000; ++i) {
        VertexPosition3Color4 vertex;
        vertex.x = d1(rng);
        vertex.y = d1(rng);
        vertex.z = d1(rng);
        auto color = Color::makeJet(d2(rng));
        vertex.r = color.r;
        vertex.g = color.g;
        vertex.b = color.b;
        vertex.a = 1.0f;
        vertices.push_back(vertex);
    }

    auto renderable = std::make_shared<PointsRenderable3>(renderer);
    renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                          window->windowSize().x);
    renderable->setPositionsAndColors(nullptr, vertices.size());
    renderable->vertexBuffer()->updateWithCuda(
        (const float*)thrust::raw_pointer_cast(vertices.data()));
    renderer->addRenderable(renderable);
}
