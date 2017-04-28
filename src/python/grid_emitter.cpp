// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_emitter.h"
#include "pybind11_utils.h"

#include <jet/grid_emitter2.h>
#include <jet/grid_emitter3.h>

namespace py = pybind11;
using namespace jet;

void addGridEmitter2(py::module& m) {
    py::class_<GridEmitter2, GridEmitter2Ptr>(m, "GridEmitter2",
        R"pbdoc(Abstract base class for 2-D grid-based emitters.)pbdoc");
}

void addGridEmitter3(py::module& m) {
    py::class_<GridEmitter3, GridEmitter3Ptr>(m, "GridEmitter3",
        R"pbdoc(Abstract base class for 3-D grid-based emitters.)pbdoc");
}
