// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "image_renderable_tests.h"

using namespace jet;
using namespace viz;

ImageRenderableTests::ImageRenderableTests(bool useOrthoCam)
    : _useOrthoCam(useOrthoCam) {}

void ImageRenderableTests::setup(GlfwWindow* window) {
    // Setup desired view controller
    if (_useOrthoCam) {
        window->setViewController(std::make_shared<OrthoViewController>(
            std::make_shared<OrthoCamera>()));
    } else {
        window->setViewController(std::make_shared<PitchYawViewController>(
            std::make_shared<PerspCamera>()));
    }

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{1, 1, 1, 1});

    // Load sample image renderable
    const ByteImage img(RESOURCES_DIR "/airplane.png");
    auto renderable = std::make_shared<ImageRenderable>(renderer);
    renderable->setImage(img);
    renderer->addRenderable(renderable);
}
