// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "particle_emitter_set.h"
#include "pybind11_utils.h"

#include <jet/particle_emitter_set2.h>
#include <jet/particle_emitter_set3.h>

namespace py = pybind11;
using namespace jet;

void addParticleEmitterSet2(py::module& m) {
    py::class_<ParticleEmitterSet2, ParticleEmitterSet2Ptr, ParticleEmitter2>(
        m, "ParticleEmitterSet2",
        R"pbdoc(
        2-D particle-based emitter set.
        )pbdoc")
        .def("__init__",
             [](ParticleEmitterSet2& instance, py::list emitters) {
                 std::vector<ParticleEmitter2Ptr> emitters_(emitters.size());
                 for (size_t i = 0; i < emitters_.size(); ++i) {
                     emitters_[i] = emitters[i].cast<ParticleEmitter2Ptr>();
                 }
                 new (&instance) ParticleEmitterSet2(emitters_);
             },
             R"pbdoc(
            Constructs an emitter with sub-emitters.
            )pbdoc",
             py::arg("emitters"))
        .def("addEmitter", &ParticleEmitterSet2::addEmitter, R"pbdoc(
            Adds sub-emitter.
            )pbdoc",
             py::arg("emitter"));
}

void addParticleEmitterSet3(py::module& m) {
    py::class_<ParticleEmitterSet3, ParticleEmitterSet3Ptr, ParticleEmitter3>(
        m, "ParticleEmitterSet3",
        R"pbdoc(
        3-D particle-based emitter set.
        )pbdoc")
        .def("__init__",
             [](ParticleEmitterSet3& instance, py::list emitters) {
                 std::vector<ParticleEmitter3Ptr> emitters_(emitters.size());
                 for (size_t i = 0; i < emitters_.size(); ++i) {
                     emitters_[i] = emitters[i].cast<ParticleEmitter3Ptr>();
                 }
                 new (&instance) ParticleEmitterSet3(emitters_);
             },
             R"pbdoc(
            Constructs an emitter with sub-emitters.
            )pbdoc",
             py::arg("emitters"))
        .def("addEmitter", &ParticleEmitterSet3::addEmitter, R"pbdoc(
            Adds sub-emitter.
            )pbdoc",
             py::arg("emitter"));
}
