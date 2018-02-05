// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "physics_animation.h"
#include "pybind11_utils.h"

#include <jet/physics_animation.h>

namespace py = pybind11;
using namespace jet;

class PyPhysicsAnimation : public PhysicsAnimation {
 public:
    using PhysicsAnimation::PhysicsAnimation;

    void onAdvanceTimeStep(double timeIntervalInSeconds) override {
        PYBIND11_OVERLOAD_PURE(void, PhysicsAnimation, onAdvanceTimeStep,
                               timeIntervalInSeconds);
    }

    unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const override {
        PYBIND11_OVERLOAD(unsigned int, PhysicsAnimation, numberOfSubTimeSteps,
                          timeIntervalInSeconds);
    }

    void onInitialize() override {
        PYBIND11_OVERLOAD(void, PhysicsAnimation, onInitialize, );
    }
};

void addPhysicsAnimation(py::module& m) {
    py::class_<PhysicsAnimation, PyPhysicsAnimation, PhysicsAnimationPtr,
               Animation>(m, "PhysicsAnimation")
        .def(py::init<>())
        .def_property("isUsingFixedSubTimeSteps",
                      &PhysicsAnimation::isUsingFixedSubTimeSteps,
                      &PhysicsAnimation::setIsUsingFixedSubTimeSteps)
        .def_property("numberOfFixedSubTimeSteps",
                      &PhysicsAnimation::numberOfFixedSubTimeSteps,
                      &PhysicsAnimation::setNumberOfFixedSubTimeSteps)
        .def("advanceSingleFrame", &PhysicsAnimation::advanceSingleFrame)
        .def_property("currentFrame", &PhysicsAnimation::currentFrame,
                      &PhysicsAnimation::setCurrentFrame)
        .def_property_readonly("currentTimeInSeconds",
                               &PhysicsAnimation::currentTimeInSeconds);
}
