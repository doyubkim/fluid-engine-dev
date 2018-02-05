// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_pressure_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_pressure_solver2.h>
#include <jet/grid_pressure_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridPressureSolver2(py::module& m) {
    py::class_<GridPressureSolver2, GridPressureSolver2Ptr>(m, "GridPressureSolver2",
        R"pbdoc(
        Abstract base class for 2-D grid-based pressure solver.

        This class represents a 2-D grid-based pressure solver interface which can
        be used as a sub-step of GridFluidSolver2.
        )pbdoc");
}

void addGridPressureSolver3(py::module& m) {
    py::class_<GridPressureSolver3, GridPressureSolver3Ptr>(m, "GridPressureSolver3",
        R"pbdoc(
        Abstract base class for 3-D grid-based pressure solver.

        This class represents a 3-D grid-based pressure solver interface which can
        be used as a sub-step of GridFluidSolver3.
        )pbdoc");
}
