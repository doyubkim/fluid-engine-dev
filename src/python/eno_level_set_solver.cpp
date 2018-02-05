// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "eno_level_set_solver.h"
#include "pybind11_utils.h"

#include <jet/eno_level_set_solver2.h>
#include <jet/eno_level_set_solver3.h>

namespace py = pybind11;
using namespace jet;

void addEnoLevelSetSolver2(py::module& m) {
    py::class_<EnoLevelSetSolver2, EnoLevelSetSolver2Ptr,
               IterativeLevelSetSolver2>(m, "EnoLevelSetSolver2",
                                         R"pbdoc(
         2-D third-order ENO-based iterative level set solver.
         )pbdoc");
}

void addEnoLevelSetSolver3(py::module& m) {
    py::class_<EnoLevelSetSolver3, EnoLevelSetSolver3Ptr,
               IterativeLevelSetSolver3>(m, "EnoLevelSetSolver3",
                                         R"pbdoc(
         3-D third-order ENO-based iterative level set solver.
         )pbdoc");
}
