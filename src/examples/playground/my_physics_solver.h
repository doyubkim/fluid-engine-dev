// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef MY_PHYSICS_SOLVER_H_
#define MY_PHYSICS_SOLVER_H_

#include <jet/jet.h>

class MyPhysicsSolver : public jet::PhysicsAnimation {
 public:
    MyPhysicsSolver();
    virtual ~MyPhysicsSolver();

 protected:
    void onInitialize() override;

    void onAdvanceTimeStep(double timeIntervalInSeconds) override;
};

#endif  // MY_PHYSICS_SOLVER_H_
