// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "sph_points_to_implicit.h"
#include "pybind11_utils.h"

#include <jet/sph_points_to_implicit2.h>
#include <jet/sph_points_to_implicit3.h>

namespace py = pybind11;
using namespace jet;

void addSphPointsToImplicit2(pybind11::module& m) {
    py::class_<SphPointsToImplicit2, PointsToImplicit2,
               SphPointsToImplicit2Ptr>(m, "SphPointsToImplicit2",
                                        R"pbdoc(
        2-D points-to-implicit converter based on standard SPH kernel.

        \see M\"uller, Matthias, David Charypar, and Markus Gross.
             "Particle-based fluid simulation for interactive applications."
             Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer
             animation. Eurographics Association, 2003.
        )pbdoc")
        .def(py::init<double, double, bool>(),
             R"pbdoc(
             Constructs the converter with given kernel radius and cut-off density.

             Parameters
             ----------
             - kernelRadius : SPH kernel radius.
             - cutOffDensity : Iso-contour value.
             - isOutputSdf : True if the output should be signed-distance field.
             )pbdoc",
             py::arg("kernelRadius") = 1.0, py::arg("cutOffDensity") = 0.5,
             py::arg("isOutputSdf") = true);
}

void addSphPointsToImplicit3(pybind11::module& m) {
    py::class_<SphPointsToImplicit3, PointsToImplicit3,
               SphPointsToImplicit3Ptr>(m, "SphPointsToImplicit3",
                                        R"pbdoc(
        3-D points-to-implicit converter based on standard SPH kernel.

        \see M\"uller, Matthias, David Charypar, and Markus Gross.
             "Particle-based fluid simulation for interactive applications."
             Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer
             animation. Eurographics Association, 2003.
        )pbdoc")
        .def(py::init<double, double, bool>(),
             R"pbdoc(
             Constructs the converter with given kernel radius and cut-off density.

             Parameters
             ----------
             - kernelRadius : SPH kernel radius.
             - cutOffDensity : Iso-contour value.
             - isOutputSdf : True if the output should be signed-distance field.
             )pbdoc",
             py::arg("kernelRadius") = 1.0, py::arg("cutOffDensity") = 0.5,
             py::arg("isOutputSdf") = true);
}
