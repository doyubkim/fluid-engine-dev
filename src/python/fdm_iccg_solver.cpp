// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_iccg_solver.h"
#include "pybind11_utils.h"

#include <jet/fdm_iccg_solver2.h>
#include <jet/fdm_iccg_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFdmIccgSolver2(py::module& m) {
    py::class_<FdmIccgSolver2, FdmIccgSolver2Ptr, FdmLinearSystemSolver2>(
        m, "FdmIccgSolver2",
        R"pbdoc(
        2-D finite difference-type linear system solver using conjugate gradient.
        )pbdoc")
        .def(py::init<uint32_t, double>(), py::arg("maxNumberOfIterations"),
             py::arg("tolerance"))
        .def_property_readonly("maxNumberOfIterations",
                               &FdmIccgSolver2::maxNumberOfIterations,
                               R"pbdoc(
            Max number of ICCG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmIccgSolver2::lastNumberOfIterations,
                               R"pbdoc(
            The last number of ICCG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmIccgSolver2::tolerance,
                               R"pbdoc(
            The max residual tolerance for the ICCG method.
            )pbdoc")
        .def_property_readonly("lastResidual", &FdmIccgSolver2::lastResidual,
                               R"pbdoc(
            The last residual after the ICCG iterations.
            )pbdoc");
}

void addFdmIccgSolver3(py::module& m) {
    py::class_<FdmIccgSolver3, FdmIccgSolver3Ptr, FdmLinearSystemSolver3>(
        m, "FdmIccgSolver3",
        R"pbdoc(
        3-D finite difference-type linear system solver using conjugate gradient.
        )pbdoc")
        .def(py::init<uint32_t, double>(), py::arg("maxNumberOfIterations"),
             py::arg("tolerance"))
        .def_property_readonly("maxNumberOfIterations",
                               &FdmIccgSolver3::maxNumberOfIterations,
                               R"pbdoc(
            Max number of ICCG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmIccgSolver3::lastNumberOfIterations,
                               R"pbdoc(
            The last number of ICCG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmIccgSolver3::tolerance,
                               R"pbdoc(
            The max residual tolerance for the ICCG method.
            )pbdoc")
        .def_property_readonly("lastResidual", &FdmIccgSolver3::lastResidual,
                               R"pbdoc(
            The last residual after the ICCG iterations.
            )pbdoc");
}
