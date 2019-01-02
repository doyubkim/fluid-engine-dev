// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <example_utils/gfx_example_manager.h>

using namespace jet;
using namespace gfx;

class SimpleExample final : public GfxExample {
 public:
    SimpleExample() : GfxExample(Frame()) {}

    std::string name() const override { return "Simple Example"; }

    void onSetup(Window* window) override {
        window->renderer()->setBackgroundColor({1.0f, 0.5f, 0.2f, 1.0f});
    }
};

class PointsExample final : public GfxExample {
 public:
    PointsExample() : GfxExample(Frame()) {}

    std::string name() const override { return "Points Example"; }

    void onSetup(Window* window) override {
        Array1<Vector3F> positions;
        Array1<Vector4F> colors;

        // Generate random points
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist1(-0.5f, 0.5f);
        std::uniform_real_distribution<float> dist2(-1.0f, 1.0f);
        for (size_t i = 0; i < 1000; ++i) {
            positions.append({dist1(rng), dist1(rng), dist1(rng)});
            colors.append(ColorUtils::makeJet(dist2(rng)));
        }

        auto pointsRenderable = std::make_shared<PointsRenderable>(
            positions, colors, 10.0f * window->displayScalingFactor().x);
        window->renderer()->addRenderable(pointsRenderable);
        window->renderer()->setBackgroundColor({0.2f, 0.5f, 1.0f, 1.0f});
    }
};

int main() {
    GlfwApp::initialize();

    auto window = GlfwApp::createWindow("OpenGL Tests", 1280, 720);

    GfxExampleManager::initialize(window);
    GfxExampleManager::addExample<SimpleExample>();
    GfxExampleManager::addExample<PointsExample>();

    return GlfwApp::run();
}
