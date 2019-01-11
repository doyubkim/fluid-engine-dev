// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "gfx_example.h"

using namespace jet;
using namespace gfx;

GfxExample::GfxExample(const Frame& frame) : _frame(frame) {}

void GfxExample::setup(Window* window) {
    _frame.index = 0;
    onSetup(window);
    onResetView(window);
}

void GfxExample::gui(Window* window) { onGui(window); }

void GfxExample::restartSim() {
    _frame.index = 0;
    onResetSim();
    onUpdateRenderables();
}

void GfxExample::advanceSim() {
    onAdvanceSim(_frame);
    onUpdateRenderables();
    ++_frame;
}

const Frame& GfxExample::currentFrame() const { return _frame; }

void GfxExample::onResetView(Window* window) {
    Viewport viewport(0, 0, window->framebufferSize().x,
                      window->framebufferSize().y);
    CameraState camera{.origin = Vector3F(0, 0, 1),
            .lookAt = Vector3F(0, 0, -1),
            .viewport = viewport};
    window->setViewController(std::make_shared<PitchYawViewController>(
            std::make_shared<PerspCamera>(camera, kHalfPiF), Vector3F()));
}

void GfxExample::onResetSim() {}

void GfxExample::onUpdateRenderables() {}

void GfxExample::onSetup(Window* window) { (void)window; }

void GfxExample::onGui(Window* window) { (void)window; }

void GfxExample::onAdvanceSim(const Frame& frame) { (void)frame; }
