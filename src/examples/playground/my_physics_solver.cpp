// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "my_physics_solver.h"

using namespace jet;

MyPhysicsSolver::MyPhysicsSolver() {}

MyPhysicsSolver::~MyPhysicsSolver() {}

void MyPhysicsSolver::onInitialize() {
    // This function is called at frame 0
    // TODO: Perform initialization here
}

void MyPhysicsSolver::onAdvanceTimeStep(double timeIntervalInSeconds) {
    // This function is called at frames greater than 0

    (void)timeIntervalInSeconds;

    // TODO: Perform simulation update here
}
