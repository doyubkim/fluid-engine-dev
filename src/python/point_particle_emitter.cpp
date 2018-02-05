// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "point_particle_emitter.h"
#include "pybind11_utils.h"

#include <jet/point_particle_emitter2.h>
#include <jet/point_particle_emitter3.h>

namespace py = pybind11;
using namespace jet;

void addPointParticleEmitter2(py::module& m) {
    py::class_<PointParticleEmitter2, PointParticleEmitter2Ptr>(
        m, "PointParticleEmitter2",
        R"pbdoc(
        2-D point particle emitter.
        )pbdoc")
        .def("__init__",
             [](PointParticleEmitter2& instance, py::object origin,
                py::object direction, double speed, double spreadAngleInDegrees,
                size_t maxNumOfNewParticlesPerSec, size_t maxNumOfParticles,
                uint32_t seed) {
                 new (&instance) PointParticleEmitter2(
                     objectToVector2D(origin), objectToVector2D(direction),
                     speed, spreadAngleInDegrees, maxNumOfNewParticlesPerSec,
                     maxNumOfParticles, seed);
             },
             R"pbdoc(
            Constructs an emitter that spawns particles from given origin,
            direction, speed, spread angle, max number of new particles per second,
            max total number of particles to be emitted, and random seed.
            )pbdoc",
             py::arg("origin"), py::arg("direction"), py::arg("speed"),
             py::arg("spreadAngleInDegrees"),
             py::arg("maxNumOfNewParticlesPerSec"),
             py::arg("maxNumOfParticles"), py::arg("seed"))
        .def_property(
            "maxNumberOfNewParticlesPerSecond",
            &PointParticleEmitter2::maxNumberOfNewParticlesPerSecond,
            &PointParticleEmitter2::setMaxNumberOfNewParticlesPerSecond,
            R"pbdoc(
            The max number of new particles per second.
            )pbdoc")
        .def_property("maxNumberOfParticles",
                      &PointParticleEmitter2::maxNumberOfParticles,
                      &PointParticleEmitter2::setMaxNumberOfParticles,
                      R"pbdoc(
            The max number of particles to be emitted.
            )pbdoc");
}

void addPointParticleEmitter3(py::module& m) {
    py::class_<PointParticleEmitter3, PointParticleEmitter3Ptr>(
        m, "PointParticleEmitter3",
        R"pbdoc(
        3-D point particle emitter.
        )pbdoc")
        .def("__init__",
             [](PointParticleEmitter3& instance, py::object origin,
                py::object direction, double speed, double spreadAngleInDegrees,
                size_t maxNumOfNewParticlesPerSec, size_t maxNumOfParticles,
                uint32_t seed) {
                 new (&instance) PointParticleEmitter3(
                     objectToVector3D(origin), objectToVector3D(direction),
                     speed, spreadAngleInDegrees, maxNumOfNewParticlesPerSec,
                     maxNumOfParticles, seed);
             },
             R"pbdoc(
            Constructs an emitter that spawns particles from given origin,
            direction, speed, spread angle, max number of new particles per second,
            max total number of particles to be emitted, and random seed.
            )pbdoc",
             py::arg("origin"), py::arg("direction"), py::arg("speed"),
             py::arg("spreadAngleInDegrees"),
             py::arg("maxNumOfNewParticlesPerSec"),
             py::arg("maxNumOfParticles"), py::arg("seed"))
        .def_property(
            "maxNumberOfNewParticlesPerSecond",
            &PointParticleEmitter3::maxNumberOfNewParticlesPerSecond,
            &PointParticleEmitter3::setMaxNumberOfNewParticlesPerSecond,
            R"pbdoc(
            The max number of new particles per second.
            )pbdoc")
        .def_property("maxNumberOfParticles",
                      &PointParticleEmitter3::maxNumberOfParticles,
                      &PointParticleEmitter3::setMaxNumberOfParticles,
                      R"pbdoc(
            The max number of particles to be emitted.
            )pbdoc");
}
