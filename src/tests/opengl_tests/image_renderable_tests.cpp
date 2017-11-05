// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "image_renderable_tests.h"

using namespace jet;
using namespace viz;

void ImageRenderableTests::setup(Renderer* renderer) {
    // Load sample image renderable
    const ByteImage img(RESOURCES_DIR "/airplane.png");
    auto renderable = std::make_shared<ImageRenderable>(renderer);
    renderable->setImage(img);
    renderer->addRenderable(renderable);
}
