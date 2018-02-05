// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_jacobi_solver.h"
#include "pybind11_utils.h"

#include <jet/fdm_jacobi_solver2.h>
#include <jet/fdm_jacobi_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFdmJacobiSolver2(py::module& m) {
    py::class_<FdmJacobiSolver2, FdmJacobiSolver2Ptr, FdmLinearSystemSolver2>(
        m, "FdmJacobiSolver2",
        R"pbdoc(
        2-D finite difference-type linear system solver using conjugate gradient.
        )pbdoc")
        .def(py::init<uint32_t, uint32_t, double>(),
             py::arg("maxNumberOfIterations"), py::arg("residualCheckInterval"),
             py::arg("tolerance"))
        .def_property_readonly("maxNumberOfIterations",
                               &FdmJacobiSolver2::maxNumberOfIterations,
                               R"pbdoc(
            Max number of CG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmJacobiSolver2::lastNumberOfIterations,
                               R"pbdoc(
            The last number of CG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmJacobiSolver2::tolerance,
                               R"pbdoc(
            The max residual tolerance for the CG method.
            )pbdoc")
        .def_property_readonly("lastResidual", &FdmJacobiSolver2::lastResidual,
                               R"pbdoc(
            The last residual after the CG iterations.
            )pbdoc");
}

void addFdmJacobiSolver3(py::module& m) {
    py::class_<FdmJacobiSolver3, FdmJacobiSolver3Ptr, FdmLinearSystemSolver3>(
        m, "FdmJacobiSolver3",
        R"pbdoc(
        3-D finite difference-type linear system solver using conjugate gradient.
        )pbdoc")
        .def(py::init<uint32_t, uint32_t, double>(),
             py::arg("maxNumberOfIterations"), py::arg("residualCheckInterval"),
             py::arg("tolerance"))
        .def_property_readonly("maxNumberOfIterations",
                               &FdmJacobiSolver3::maxNumberOfIterations,
                               R"pbdoc(
            Max number of CG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmJacobiSolver3::lastNumberOfIterations,
                               R"pbdoc(
            The last number of CG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmJacobiSolver3::tolerance,
                               R"pbdoc(
            The max residual tolerance for the CG method.
            )pbdoc")
        .def_property_readonly("lastResidual", &FdmJacobiSolver3::lastResidual,
                               R"pbdoc(
            The last residual after the CG iterations.
            )pbdoc");
}
