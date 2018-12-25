// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "gfx_example.h"

GfxExample::GfxExample(const jet::Frame& frame) : _frame(frame) {}

void GfxExample::setup(jet::gfx::Window* window) {
    _frame.index = 0;
    onSetup(window);
}

void GfxExample::gui(jet::gfx::Window* window) { onGui(window); }

void GfxExample::restartSim() {
    _frame.index = 0;
    onRestartSim();
}

void GfxExample::advanceSim() {
    onAdvanceSim(_frame);
    ++_frame;
}

void GfxExample::onRestartSim() {}

void GfxExample::onSetup(jet::gfx::Window* window) { (void)window; }

void GfxExample::onGui(jet::gfx::Window* window) { (void)window; }

void GfxExample::onAdvanceSim(const jet::Frame& frame) { (void)frame; }
