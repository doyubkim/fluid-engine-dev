// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "simple_volume_renderable_tests.h"

#include <imgui/imgui.h>

#include <jet/jet.h>

using namespace jet;
using namespace viz;

void SimpleVolumeRenderableTests::setup(GlfwWindow* window) {
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
        const float phi0 = (grid - Vector3F(32, 32, 32)).length() - 12.0f;
        const float phi1 = (grid - Vector3F(20, 40, 30)).length() - 8.0f;
        const float phi2 = (grid - Vector3F(40, 30, 15)).length() - 4.0f;

        const float phi = min3(phi0, phi1, phi2);

        const float den = 1.0f - smearedHeavisideSdf(phi / 10.0f);
        data(i, j, k).r = (float)i / 64;
        data(i, j, k).g = (float)j / 64;
        data(i, j, k).b = (float)k / 64;
        data(i, j, k).a = den;

    });

    _renderable = std::make_shared<SimpleVolumeRenderable>(renderer);
    _renderable->setVolume(data.constAccessor());
    renderer->addRenderable(_renderable);
}

void SimpleVolumeRenderableTests::onGui(GlfwWindow*) {
    bool needsUpdate = false;

    ImGui::Begin("Parameters");
    {
        float brightness = _renderable->brightness();
        ImGui::SliderFloat("Brightness", &brightness, 0.0f, 1.0f, "%.2f");
        needsUpdate |= brightness != _renderable->brightness();
        _renderable->setBrightness(brightness);

        float density = _renderable->density();
        ImGui::SliderFloat("Density", &density, 0.0f, 1.0f, "%.2f");
        needsUpdate |= density != _renderable->density();
        _renderable->setDensity(density);

        float stepSize = _renderable->stepSize();
        ImGui::SliderFloat("Step size", &stepSize, 0.001f, 0.050f, "%.3f");
        needsUpdate |= stepSize != _renderable->stepSize();
        _renderable->setStepSize(stepSize);
    }
    ImGui::End();

    if (needsUpdate) {
        _renderable->requestUpdateVertexBuffer();
    }
}