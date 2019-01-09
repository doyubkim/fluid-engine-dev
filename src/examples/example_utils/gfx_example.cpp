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
}

void GfxExample::gui(Window* window) { onGui(window); }

void GfxExample::restartSim() {
    _frame.index = 0;
    onRestartSim();
}

void GfxExample::advanceSim() {
    onAdvanceSim(_frame);
    ++_frame;
}

const Frame& GfxExample::currentFrame() const { return _frame; }

void GfxExample::onRestartSim() {}

void GfxExample::onSetup(Window* window) { (void)window; }

void GfxExample::onGui(Window* window) { (void)window; }

void GfxExample::onAdvanceSim(const Frame& frame) { (void)frame; }
