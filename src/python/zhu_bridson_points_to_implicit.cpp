// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "zhu_bridson_points_to_implicit.h"
#include "pybind11_utils.h"

#include <jet/zhu_bridson_points_to_implicit2.h>
#include <jet/zhu_bridson_points_to_implicit3.h>

namespace py = pybind11;
using namespace jet;

void addZhuBridsonPointsToImplicit2(pybind11::module& m) {
    py::class_<ZhuBridsonPointsToImplicit2, PointsToImplicit2,
               ZhuBridsonPointsToImplicit2Ptr>(m, "ZhuBridsonPointsToImplicit2",
                                               R"pbdoc(
        2-D points-to-implicit converter based on Zhu and Bridson's method.

        \see Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid."
             ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
        )pbdoc")
        .def(py::init<double, double, bool>(),
             R"pbdoc(
             Constructs the converter with given kernel radius and cut-off threshold.

             Parameters
             ----------
             - kernelRadius : Smoothing kernel radius.
             - cutOffThreshold : Iso-contour value.
             - isOutputSdf : True if the output should be signed-distance field.
             )pbdoc",
             py::arg("kernelRadius") = 1.0, py::arg("cutOffThreshold") = 0.25,
             py::arg("isOutputSdf") = true);
}

void addZhuBridsonPointsToImplicit3(pybind11::module& m) {
    py::class_<ZhuBridsonPointsToImplicit3, PointsToImplicit3,
               ZhuBridsonPointsToImplicit3Ptr>(m, "ZhuBridsonPointsToImplicit3",
                                               R"pbdoc(
        3-D points-to-implicit converter based on Zhu and Bridson's method.

        \see Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid."
             ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
        )pbdoc")
        .def(py::init<double, double, bool>(),
             R"pbdoc(
             Constructs the converter with given kernel radius and cut-off threshold.

             Parameters
             ----------
             - kernelRadius : Smoothing kernel radius.
             - cutOffThreshold : Iso-contour value.
             - isOutputSdf : True if the output should be signed-distance field.
             )pbdoc",
             py::arg("kernelRadius") = 1.0, py::arg("cutOffThreshold") = 0.25,
             py::arg("isOutputSdf") = true);
}
