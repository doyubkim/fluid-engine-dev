// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "anisotropic_points_to_implicit.h"
#include "pybind11_utils.h"

#include <jet/anisotropic_points_to_implicit2.h>
#include <jet/anisotropic_points_to_implicit3.h>

namespace py = pybind11;
using namespace jet;

void addAnisotropicPointsToImplicit2(pybind11::module& m) {
    py::class_<AnisotropicPointsToImplicit2, PointsToImplicit2,
               AnisotropicPointsToImplicit2Ptr>(m,
                                                "AnisotropicPointsToImplicit2",
                                                R"pbdoc(
        2-D points-to-implicit converter using Anisotropic kernels.

        \see Yu, Jihun, and Greg Turk. "Reconstructing surfaces of particle-based
             fluids using anisotropic kernels." ACM Transactions on Graphics (TOG)
             32.1 (2013): 5.
        )pbdoc")
        .def(py::init<double, double, double, size_t, bool>(),
             R"pbdoc(
             Constructs the converter with given kernel radius and cut-off density.

             Parameters
             ----------
             - kernelRadius : Smoothing kernel radius.
             - cutOffDensity : Iso-contour value.
             - positionSmoothingFactor : Position smoothing factor.
             - minNumNeighbors : Minimum number of neighbors to enable anisotropic kernel.
             - isOutputSdf : True if the output should be signed-distance field.
             )pbdoc",
             py::arg("kernelRadius") = 1.0, py::arg("cutOffDensity") = 0.5,
             py::arg("positionSmoothingFactor") = 0.5,
             py::arg("minNumNeighbors") = 8, py::arg("isOutputSdf") = true);
}

void addAnisotropicPointsToImplicit3(pybind11::module& m) {
    py::class_<AnisotropicPointsToImplicit3, PointsToImplicit3,
               AnisotropicPointsToImplicit3Ptr>(m,
                                                "AnisotropicPointsToImplicit3",
                                                R"pbdoc(
        3-D points-to-implicit converter using Anisotropic kernels.

        \see Yu, Jihun, and Greg Turk. "Reconstructing surfaces of particle-based
             fluids using anisotropic kernels." ACM Transactions on Graphics (TOG)
             32.1 (2013): 5.
        )pbdoc")
        .def(py::init<double, double, double, size_t, bool>(),
             R"pbdoc(
             Constructs the converter with given kernel radius and cut-off density.

             Parameters
             ----------
             - kernelRadius : Smoothing kernel radius.
             - cutOffDensity : Iso-contour value.
             - positionSmoothingFactor : Position smoothing factor.
             - minNumNeighbors : Minimum number of neighbors to enable anisotropic kernel.
             - isOutputSdf : True if the output should be signed-distance field.
             )pbdoc",
             py::arg("kernelRadius") = 1.0, py::arg("cutOffDensity") = 0.5,
             py::arg("positionSmoothingFactor") = 0.5,
             py::arg("minNumNeighbors") = 25, py::arg("isOutputSdf") = true);
}
