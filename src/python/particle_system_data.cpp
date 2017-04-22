// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "particle_system_data.h"
#include "pybind11_utils.h"

#include <jet/particle_system_data3.h>

namespace py = pybind11;
using namespace jet;

void addParticleSystemData3(py::module& m) {
    py::class_<ParticleSystemData3, ParticleSystemData3Ptr>(
        m, "ParticleSystemData3")
        .def("__init__",
             [](ParticleSystemData3& instance, py::args args,
                py::kwargs kwargs) {
                 size_t numberOfParticles = 0;

                 if (args.size() == 1) {
                     if (py::isinstance<ParticleSystemData3>(args[0])) {
                         new (&instance) ParticleSystemData3(
                             *args[0].cast<ParticleSystemData3Ptr>());
                         return;
                     } else {
                         numberOfParticles = args[0].cast<size_t>();
                     }
                 }

                 if (kwargs.contains("numberOfParticles")) {
                     numberOfParticles =
                         kwargs["numberOfParticles"].cast<size_t>();
                 }

                 new (&instance) ParticleSystemData3(numberOfParticles);
             })
        .def_property_readonly("numberOfParticles",
                               &ParticleSystemData3::numberOfParticles);
}
