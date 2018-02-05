// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "level_set_solver.h"
#include "pybind11_utils.h"

#include <jet/level_set_solver2.h>
#include <jet/level_set_solver3.h>

namespace py = pybind11;
using namespace jet;

void addLevelSetSolver2(py::module& m) {
    py::class_<LevelSetSolver2, LevelSetSolver2Ptr>(m, "LevelSetSolver2",
                                                    R"pbdoc(
         Abstract base class for 2-D level set solver.
         )pbdoc");
}

void addLevelSetSolver3(py::module& m) {
    py::class_<LevelSetSolver3, LevelSetSolver3Ptr>(m, "LevelSetSolver3",
                                                    R"pbdoc(
         Abstract base class for 3-D level set solver.
         )pbdoc");
}
