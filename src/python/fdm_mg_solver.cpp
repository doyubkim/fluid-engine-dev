// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_mg_solver.h"
#include "pybind11_utils.h"

#include <jet/fdm_mg_solver2.h>

namespace py = pybind11;
using namespace jet;

void addFdmMgSolver2(py::module& m) {
    py::class_<FdmMgSolver2, FdmMgSolver2Ptr, FdmLinearSystemSolver2>(
        m, "FdmMgSolver2",
        R"pbdoc(
        2-D finite difference-type linear system solver using multigrid.
        )pbdoc")
        .def(py::init<size_t, uint32_t, uint32_t, uint32_t, uint32_t, double>(),
             py::arg("maxNumberOfLevels"),
             py::arg("numberOfRestrictionIter") = 10,
             py::arg("numberOfCorrectionIter") = 10,
             py::arg("numberOfCoarsestIter") = 10,
             py::arg("numberOfFinalIter") = 10, py::arg("maxTolerance") = 1e-9)
        .def_property_readonly("maxNumberOfLevels",
                               [](const FdmMgSolver2& instance) {
                                   return instance.params().maxNumberOfLevels;
                               },
                               R"pbdoc(
            Max number of multigrid levels.
            )pbdoc")
        .def_property_readonly(
            "numberOfRestrictionIter",
            [](const FdmMgSolver2& instance) {
                return instance.params().numberOfRestrictionIter;
            },
            R"pbdoc(
            Number of iteration at restriction step.
            )pbdoc")
        .def_property_readonly(
            "numberOfCorrectionIter",
            [](const FdmMgSolver2& instance) {
                return instance.params().numberOfCorrectionIter;
            },
            R"pbdoc(
            Number of iteration at correction step.
            )pbdoc")
        .def_property_readonly(
            "numberOfCoarsestIter",
            [](const FdmMgSolver2& instance) {
                return instance.params().numberOfCoarsestIter;
            },
            R"pbdoc(
            Number of iteration at coarsest step.
            )pbdoc")
        .def_property_readonly("numberOfFinalIter",
                               [](const FdmMgSolver2& instance) {
                                   return instance.params().numberOfFinalIter;
                               },
                               R"pbdoc(
            Number of iteration at final step.
            )pbdoc");
}
