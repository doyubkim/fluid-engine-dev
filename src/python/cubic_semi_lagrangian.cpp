// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cubic_semi_lagrangian.h"
#include "pybind11_utils.h"

#include <jet/cubic_semi_lagrangian2.h>
#include <jet/cubic_semi_lagrangian3.h>

namespace py = pybind11;
using namespace jet;

void addCubicSemiLagrangian2(py::module& m) {
    py::class_<CubicSemiLagrangian2, CubicSemiLagrangian2Ptr, SemiLagrangian2>(
        m, "CubicSemiLagrangian2",
        R"pbdoc(
        Implementation of 2-D cubic semi-Lagrangian advection solver.

        This class implements 3rd-order cubic 2-D semi-Lagrangian advection solver.
        )pbdoc")
        .def(py::init<>());
}

void addCubicSemiLagrangian3(py::module& m) {
    py::class_<CubicSemiLagrangian3, CubicSemiLagrangian3Ptr, SemiLagrangian3>(
        m, "CubicSemiLagrangian3",
        R"pbdoc(
        Implementation of 3-D cubic semi-Lagrangian advection solver.

        This class implements 3rd-order cubic 3-D semi-Lagrangian advection solver.
        )pbdoc")
        .def(py::init<>());
}
