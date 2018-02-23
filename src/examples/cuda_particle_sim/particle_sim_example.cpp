// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "particle_sim_example.h"

ParticleSimExample::ParticleSimExample(const jet::Frame& frame)
    : _frame(frame) {}

void ParticleSimExample::setup(jet::viz::GlfwWindow* window) {
    onSetup(window);
}

void ParticleSimExample::gui(jet::viz::GlfwWindow* window) { onGui(window); }

void ParticleSimExample::update() {
    onUpdate(_frame);
    ++_frame;
}

void ParticleSimExample::onSetup(jet::viz::GlfwWindow* window) { (void)window; }

void ParticleSimExample::onGui(jet::viz::GlfwWindow* window) { (void)window; }

void ParticleSimExample::onUpdate(const jet::Frame& frame) { (void)frame; }
