// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "example.h"

Example::Example(const jet::Frame& frame) : _frame(frame) {}

void Example::setup(jet::viz::GlfwWindow* window) { onSetup(window); }

void Example::gui(jet::viz::GlfwWindow* window) { onGui(window); }

void Example::advanceSim() {
    onAdvanceSim(_frame);
    ++_frame;
}

void Example::updateRenderables() {
    onUpdateRenderables();
}

void Example::onSetup(jet::viz::GlfwWindow* window) { (void)window; }

void Example::onGui(jet::viz::GlfwWindow* window) { (void)window; }

void Example::onAdvanceSim(const jet::Frame& frame) { (void)frame; }

void Example::onUpdateRenderables() {}
