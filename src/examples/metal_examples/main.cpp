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

int main() {
    MetalApp::initialize();

    auto window = MetalApp::createWindow("Metal Tests", 1280, 720);

    GfxExampleManager::initialize(window);
    GfxExampleManager::addExample<SimpleExample>();

    return MetalApp::run();
}
