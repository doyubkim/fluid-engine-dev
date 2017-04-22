// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "particle_emitter.h"
#include "pybind11_utils.h"

#include <jet/particle_emitter3.h>

namespace py = pybind11;
using namespace jet;

void addParticleEmitter3(py::module& m) {
    py::class_<ParticleEmitter3, ParticleEmitter3Ptr>(m, "ParticleEmitter3");
}
