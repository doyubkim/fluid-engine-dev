// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "simple_volume_renderable_tests.h"

#include <jet/jet.h>

using namespace jet;
using namespace viz;

void SimpleVolumeRenderableTests::setup(GLFWWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<SphericalViewController>(
        std::make_shared<PerspCamera>()));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{0, 0, 0, 1});

    // Generate sphere
    Array3<Color> data(64, 64, 64);
    data.forEachIndex([&](size_t i, size_t j, size_t k) {
        const Vector3F grid((float)i, (float)j, (float)k);
        const float phi = (grid - Vector3F(32, 32, 32)).length() - 12.0f;
        const float den = 1.0f - smearedHeavisideSdf(phi / 10.0f);
        data(i, j, k).r = 1.0f;
        data(i, j, k).g = 1.0f;
        data(i, j, k).b = 1.0f;
        data(i, j, k).a = den;

    });

    auto renderable = std::make_shared<SimpleVolumeRenderable>(renderer);
    renderable->setVolume(data.data(), Size3{64, 64, 64});
    renderer->addRenderable(renderable);
}
