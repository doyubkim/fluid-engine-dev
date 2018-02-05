// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_gauss_seidel_solver.h"
#include "pybind11_utils.h"

#include <jet/fdm_gauss_seidel_solver2.h>
#include <jet/fdm_gauss_seidel_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFdmGaussSeidelSolver2(py::module& m) {
    py::class_<FdmGaussSeidelSolver2, FdmGaussSeidelSolver2Ptr,
               FdmLinearSystemSolver2>(m, "FdmGaussSeidelSolver2",
                                       R"pbdoc(
        2-D finite difference-type linear system solver using conjugate gradient.
        )pbdoc")
        .def(py::init<uint32_t, uint32_t, double, double, bool>(),
             py::arg("maxNumberOfIterations"), py::arg("residualCheckInterval"),
             py::arg("tolerance"), py::arg("sorFactor") = 1.0,
             py::arg("useRedBlackOrdering") = false)
        .def_property_readonly("maxNumberOfIterations",
                               &FdmGaussSeidelSolver2::maxNumberOfIterations,
                               R"pbdoc(
            Max number of CG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmGaussSeidelSolver2::lastNumberOfIterations,
                               R"pbdoc(
            The last number of CG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmGaussSeidelSolver2::tolerance,
                               R"pbdoc(
            The max residual tolerance for the CG method.
            )pbdoc")
        .def_property_readonly("lastResidual",
                               &FdmGaussSeidelSolver2::lastResidual,
                               R"pbdoc(
            The last residual after the CG iterations.
            )pbdoc")
        .def_property_readonly("sorFactor", &FdmGaussSeidelSolver2::sorFactor,
                               R"pbdoc(
            Returns the SOR (Successive Over Relaxation) factor.
            )pbdoc")
        .def_property_readonly("useRedBlackOrdering",
                               &FdmGaussSeidelSolver2::useRedBlackOrdering,
                               R"pbdoc(
            Returns true if red-black ordering is enabled.
            )pbdoc");
}

void addFdmGaussSeidelSolver3(py::module& m) {
    py::class_<FdmGaussSeidelSolver3, FdmGaussSeidelSolver3Ptr,
               FdmLinearSystemSolver3>(m, "FdmGaussSeidelSolver3",
                                       R"pbdoc(
        3-D finite difference-type linear system solver using conjugate gradient.
        )pbdoc")
        .def(py::init<uint32_t, uint32_t, double, double, bool>(),
             py::arg("maxNumberOfIterations"), py::arg("residualCheckInterval"),
             py::arg("tolerance"), py::arg("sorFactor") = 1.0,
             py::arg("useRedBlackOrdering") = false)
        .def_property_readonly("maxNumberOfIterations",
                               &FdmGaussSeidelSolver3::maxNumberOfIterations,
                               R"pbdoc(
            Max number of CG iterations.
            )pbdoc")
        .def_property_readonly("lastNumberOfIterations",
                               &FdmGaussSeidelSolver3::lastNumberOfIterations,
                               R"pbdoc(
            The last number of CG iterations the solver made.
            )pbdoc")
        .def_property_readonly("tolerance", &FdmGaussSeidelSolver3::tolerance,
                               R"pbdoc(
            The max residual tolerance for the CG method.
            )pbdoc")
        .def_property_readonly("lastResidual",
                               &FdmGaussSeidelSolver3::lastResidual,
                               R"pbdoc(
            The last residual after the CG iterations.
            )pbdoc")
        .def_property_readonly("sorFactor", &FdmGaussSeidelSolver3::sorFactor,
                               R"pbdoc(
            Returns the SOR (Successive Over Relaxation) factor.
            )pbdoc")
        .def_property_readonly("useRedBlackOrdering",
                               &FdmGaussSeidelSolver3::useRedBlackOrdering,
                               R"pbdoc(
            Returns true if red-black ordering is enabled.
            )pbdoc");
}
