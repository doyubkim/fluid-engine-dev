// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <example_utils/gfx_demo.h>

using namespace jet;
using namespace gfx;

int main() {
    MetalApp::initialize();

    auto window = MetalApp::createWindow("Metal Tests", 1280, 720);
    makeGfxDemo(window);

    return MetalApp::run();
}
