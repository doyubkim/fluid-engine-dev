// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_mgpcg_solver.h"
#include "pybind11_utils.h"

#include <jet/fdm_mgpcg_solver2.h>

namespace py = pybind11;
using namespace jet;

void addFdmMgpcgSolver2(py::module& m) {
    py::class_<FdmMgpcgSolver2, FdmMgpcgSolver2Ptr, FdmLinearSystemSolver2>(
        m, "FdmMgpcgSolver2",
        R"pbdoc(
        2-D finite difference-type linear system solver using MGPCG.
        )pbdoc")
        .def(py::init<uint32_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t,
                      double>(),
             py::arg("numberOfCgIter"), py::arg("maxNumberOfLevels"),
             py::arg("numberOfRestrictionIter") = 10,
             py::arg("numberOfCorrectionIter") = 10,
             py::arg("numberOfCoarsestIter") = 10,
             py::arg("numberOfFinalIter") = 10, py::arg("maxTolerance") = 1e-9)
        .def_property_readonly("maxNumberOfIterations",
                               &FdmMgpcgSolver2::maxNumberOfIterations,
                               R"pbdoc(
            Max number of MGPCG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmMgpcgSolver2::lastNumberOfIterations,
                               R"pbdoc(
            The last number of MGPCG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmMgpcgSolver2::tolerance,
                               R"pbdoc(
            The max residual tolerance for the MGPCG method.
            )pbdoc")
        .def_property_readonly("lastResidual", &FdmMgpcgSolver2::lastResidual,
                               R"pbdoc(
            The last residual after the MGPCG iterations.
            )pbdoc");
}
