// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "particle_emitter.h"
#include "pybind11_utils.h"

#include <jet/particle_emitter2.h>
#include <jet/particle_emitter3.h>

namespace py = pybind11;
using namespace jet;

void addParticleEmitter2(py::module& m) {
    py::class_<ParticleEmitter2, ParticleEmitter2Ptr>(m, "ParticleEmitter2",
                                                      R"pbdoc(
        Abstract base class for 2-D particle emitter.
        )pbdoc")
        .def("update", &ParticleEmitter2::update, R"pbdoc(
            Updates the emitter state from `currentTimeInSeconds` to the following
            time-step.
            )pbdoc",
             py::arg("currentTimeInSeconds"), py::arg("timeIntervalInSeconds"))
        .def_property("target", &ParticleEmitter2::target,
                      &ParticleEmitter2::setTarget, R"pbdoc(
            The target particle system to emit.
            )pbdoc")
        .def_property(
            "isEnabled", &ParticleEmitter2::isEnabled,
            &ParticleEmitter2::setIsEnabled,
            R"pbdoc(True/false if the emitter is enabled/disabled.)pbdoc")
        .def("setOnBeginUpdateCallback",
             &ParticleEmitter2::setOnBeginUpdateCallback,
             R"pbdoc(
            Sets the callback function to be called when `update` function is invoked.

            The callback function takes current simulation time in seconds unit. Use
            this callback to track any motion or state changes related to this
            emitter.
            )pbdoc");
}

void addParticleEmitter3(py::module& m) {
    py::class_<ParticleEmitter3, ParticleEmitter3Ptr>(m, "ParticleEmitter3",
                                                      R"pbdoc(
        Abstract base class for 3-D particle emitter.
        )pbdoc")
        .def("update", &ParticleEmitter3::update, R"pbdoc(
            Updates the emitter state from `currentTimeInSeconds` to the following
            time-step.
            )pbdoc",
             py::arg("currentTimeInSeconds"), py::arg("timeIntervalInSeconds"))
        .def_property("target", &ParticleEmitter3::target,
                      &ParticleEmitter3::setTarget, R"pbdoc(
            The target particle system to emit.
            )pbdoc")
        .def_property(
            "isEnabled", &ParticleEmitter3::isEnabled,
            &ParticleEmitter3::setIsEnabled,
            R"pbdoc(True/false if the emitter is enabled/disabled.)pbdoc")
        .def("setOnBeginUpdateCallback",
             &ParticleEmitter3::setOnBeginUpdateCallback,
             R"pbdoc(
            Sets the callback function to be called when `update` function is invoked.

            The callback function takes current simulation time in seconds unit. Use
            this callback to track any motion or state changes related to this
            emitter.
            )pbdoc");
}
