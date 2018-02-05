// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "spherical_points_to_implicit.h"
#include "pybind11_utils.h"

#include <jet/spherical_points_to_implicit2.h>
#include <jet/spherical_points_to_implicit3.h>

namespace py = pybind11;
using namespace jet;

void addSphericalPointsToImplicit2(pybind11::module& m) {
    py::class_<SphericalPointsToImplicit2, PointsToImplicit2,
               SphericalPointsToImplicit2Ptr>(m, "SphericalPointsToImplicit2",
                                              R"pbdoc(
        2-D points-to-implicit converter based on simple sphere model.
        )pbdoc")
        .def(py::init<double, bool>(),
             R"pbdoc(
             Constructs the converter with given sphere radius.

             Parameters
             ----------
             - radius : Sphere radius.
             - isOutputSdf : True if the output should be signed-distance field.
             )pbdoc",
             py::arg("radius") = 1.0, py::arg("isOutputSdf") = true);
}

void addSphericalPointsToImplicit3(pybind11::module& m) {
    py::class_<SphericalPointsToImplicit3, PointsToImplicit3,
               SphericalPointsToImplicit3Ptr>(m, "SphericalPointsToImplicit3",
                                              R"pbdoc(
        3-D points-to-implicit converter based on simple sphere model.
        )pbdoc")
        .def(py::init<double, bool>(),
             R"pbdoc(
             Constructs the converter with given sphere radius.

             Parameters
             ----------
             - radius : Sphere radius.
             - isOutputSdf : True if the output should be signed-distance field.
             )pbdoc",
             py::arg("radius") = 1.0, py::arg("isOutputSdf") = true);
}
