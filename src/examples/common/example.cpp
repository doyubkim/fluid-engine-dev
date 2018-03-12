// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "example.h"

Example::Example(const jet::Frame& frame) : _frame(frame) {}

#ifdef JET_USE_GL
void Example::setup(jet::viz::GlfwWindow* window) {
    _frame.index = 0;
    onSetup(window);
}

void Example::gui(jet::viz::GlfwWindow* window) { onGui(window); }
#else
void Example::setup() { onSetup(); }
#endif

void Example::restartSim() {
    _frame.index = 0;
    onRestartSim();
}

void Example::advanceSim() {
    onAdvanceSim(_frame);
    ++_frame;
}

void Example::updateRenderables() { onUpdateRenderables(); }

void Example::onRestartSim() {}

#ifdef JET_USE_GL
void Example::onSetup(jet::viz::GlfwWindow* window) { (void)window; }

void Example::onGui(jet::viz::GlfwWindow* window) { (void)window; }
#else
void Example::onSetup() {}
#endif

void Example::onAdvanceSim(const jet::Frame& frame) { (void)frame; }

void Example::onUpdateRenderables() {}
