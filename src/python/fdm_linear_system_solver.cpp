// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_linear_system_solver.h"
#include "pybind11_utils.h"

#include <jet/fdm_linear_system_solver2.h>
#include <jet/fdm_linear_system_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFdmLinearSystemSolver2(py::module& m) {
    py::class_<FdmLinearSystemSolver2, FdmLinearSystemSolver2Ptr>(
        m, "FdmLinearSystemSolver2",
        R"pbdoc(
        Abstract base class for 2-D finite difference-type linear system solver.
        )pbdoc");
}

void addFdmLinearSystemSolver3(py::module& m) {
    py::class_<FdmLinearSystemSolver3, FdmLinearSystemSolver3Ptr>(
        m, "FdmLinearSystemSolver3",
        R"pbdoc(
        Abstract base class for 3-D finite difference-type linear system solver.
        )pbdoc");
}
