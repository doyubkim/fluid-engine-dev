// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "upwind_level_set_solver.h"
#include "pybind11_utils.h"

#include <jet/upwind_level_set_solver2.h>
#include <jet/upwind_level_set_solver3.h>

namespace py = pybind11;
using namespace jet;

void addUpwindLevelSetSolver2(py::module& m) {
    py::class_<UpwindLevelSetSolver2, UpwindLevelSetSolver2Ptr,
               IterativeLevelSetSolver2>(m, "UpwindLevelSetSolver2",
                                         R"pbdoc(
         2-D first-order upwind-based iterative level set solver.
         )pbdoc");
}

void addUpwindLevelSetSolver3(py::module& m) {
    py::class_<UpwindLevelSetSolver3, UpwindLevelSetSolver3Ptr,
               IterativeLevelSetSolver3>(m, "UpwindLevelSetSolver3",
                                         R"pbdoc(
         3-D first-order upwind-based iterative level set solver.
         )pbdoc");
}
