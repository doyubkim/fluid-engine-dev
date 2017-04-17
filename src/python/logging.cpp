// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "logging.h"
#include "pybind11_utils.h"

#include <jet/logging.h>

namespace py = pybind11;
using namespace jet;

void addLogging(pybind11::module& m) {
    py::class_<Logging>(m, "Logging")
        .def_static("mute", &Logging::mute)
        .def_static("unmute", &Logging::unmute);
}
