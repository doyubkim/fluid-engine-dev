// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pci_sph_solver.h"
#include "pybind11_utils.h"

#include <jet/pci_sph_solver2.h>
#include <jet/pci_sph_solver3.h>

namespace py = pybind11;
using namespace jet;

void addPciSphSolver2(py::module& m) {
    py::class_<PciSphSolver2, PciSphSolver2Ptr, SphSolver2>(m, "PciSphSolver2",
                                                            R"pbdoc(
        2-D PCISPH solver.

        This class implements 2-D predictive-corrective SPH solver. The main
        pressure solver is based on Solenthaler and Pajarola's 2009 SIGGRAPH paper.
        - See Solenthaler and Pajarola, Predictive-corrective incompressible SPH,
              ACM transactions on graphics (TOG). Vol. 28. No. 3. ACM, 2009.
        )pbdoc")
        .def(py::init<double, double, double>(),
             R"pbdoc(
             Constructs a solver with target density, spacing, and relative kernel
             radius.
             )pbdoc",
             py::arg("targetDensity") = kWaterDensity,
             py::arg("targetSpacing") = 0.1,
             py::arg("relativeKernelRadius") = 1.8)
        .def_property("maxDensityErrorRatio",
                      &PciSphSolver2::maxDensityErrorRatio,
                      &PciSphSolver2::setMaxDensityErrorRatio,
                      R"pbdoc(
             The max allowed density error ratio.

             This property sets the max allowed density error ratio during the PCISPH
             iteration. Default is 0.01 (1%). The input value should be positive.
             )pbdoc")
        .def_property("maxNumberOfIterations",
                      &PciSphSolver2::maxNumberOfIterations,
                      &PciSphSolver2::setMaxNumberOfIterations,
                      R"pbdoc(
             The max number of PCISPH iterations.

             This property sets the max number of PCISPH iterations. Default is 5.
             )pbdoc");
}

void addPciSphSolver3(py::module& m) {
    py::class_<PciSphSolver3, PciSphSolver3Ptr, SphSolver3>(m, "PciSphSolver3",
                                                            R"pbdoc(
        3-D PCISPH solver.

        This class implements 3-D predictive-corrective SPH solver. The main
        pressure solver is based on Solenthaler and Pajarola's 2009 SIGGRAPH paper.
        - See Solenthaler and Pajarola, Predictive-corrective incompressible SPH,
              ACM transactions on graphics (TOG). Vol. 28. No. 3. ACM, 2009.
        )pbdoc")
        .def(py::init<double, double, double>(),
             R"pbdoc(
             Constructs a solver with target density, spacing, and relative kernel
             radius.
             )pbdoc",
             py::arg("targetDensity") = kWaterDensity,
             py::arg("targetSpacing") = 0.1,
             py::arg("relativeKernelRadius") = 1.8)
        .def_property("maxDensityErrorRatio",
                      &PciSphSolver3::maxDensityErrorRatio,
                      &PciSphSolver3::setMaxDensityErrorRatio,
                      R"pbdoc(
             The max allowed density error ratio.

             This property sets the max allowed density error ratio during the PCISPH
             iteration. Default is 0.01 (1%). The input value should be positive.
             )pbdoc")
        .def_property("maxNumberOfIterations",
                      &PciSphSolver3::maxNumberOfIterations,
                      &PciSphSolver3::setMaxNumberOfIterations,
                      R"pbdoc(
             The max number of PCISPH iterations.

             This property sets the max number of PCISPH iterations. Default is 5.
             )pbdoc");
}
