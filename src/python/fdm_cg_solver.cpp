// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_cg_solver.h"
#include "pybind11_utils.h"

#include <jet/fdm_cg_solver2.h>
#include <jet/fdm_cg_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFdmCgSolver2(py::module& m) {
    py::class_<FdmCgSolver2, FdmCgSolver2Ptr, FdmLinearSystemSolver2>(
        m, "FdmCgSolver2",
        R"pbdoc(
        2-D finite difference-type linear system solver using conjugate gradient.
        )pbdoc")
        .def(py::init<uint32_t, double>(), py::arg("maxNumberOfIterations"),
             py::arg("tolerance"))
        .def_property_readonly("maxNumberOfIterations",
                               &FdmCgSolver2::maxNumberOfIterations,
                               R"pbdoc(
            Max number of CG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmCgSolver2::lastNumberOfIterations,
                               R"pbdoc(
            The last number of CG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmCgSolver2::tolerance,
                               R"pbdoc(
            The max residual tolerance for the CG method.
            )pbdoc")
        .def_property_readonly("lastResidual", &FdmCgSolver2::lastResidual,
                               R"pbdoc(
            The last residual after the CG iterations.
            )pbdoc");
}

void addFdmCgSolver3(py::module& m) {
    py::class_<FdmCgSolver3, FdmCgSolver3Ptr, FdmLinearSystemSolver3>(
        m, "FdmCgSolver3",
        R"pbdoc(
        3-D finite difference-type linear system solver using conjugate gradient.
        )pbdoc")
        .def(py::init<uint32_t, double>(), py::arg("maxNumberOfIterations"),
             py::arg("tolerance"))
        .def_property_readonly("maxNumberOfIterations",
                               &FdmCgSolver3::maxNumberOfIterations,
                               R"pbdoc(
            Max number of CG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmCgSolver3::lastNumberOfIterations,
                               R"pbdoc(
            The last number of CG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmCgSolver3::tolerance,
                               R"pbdoc(
            The max residual tolerance for the CG method.
            )pbdoc")
        .def_property_readonly("lastResidual", &FdmCgSolver3::lastResidual,
                               R"pbdoc(
            The last residual after the CG iterations.
            )pbdoc");
}
