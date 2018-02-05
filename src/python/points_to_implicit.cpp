// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "points_to_implicit.h"
#include "pybind11_utils.h"

#include <jet/points_to_implicit2.h>
#include <jet/points_to_implicit3.h>

namespace py = pybind11;
using namespace jet;

void addPointsToImplicit2(pybind11::module& m) {
    py::class_<PointsToImplicit2, PointsToImplicit2Ptr>(m, "PointsToImplicit2",
                                                        R"pbdoc(
        Abstract base class for 2-D points-to-implicit converters.
        )pbdoc")
        .def("convert",
             [](const PointsToImplicit2& instance, const py::list& points,
                ScalarGrid2Ptr output) {
                 std::vector<Vector2D> points_;
                 for (size_t i = 0; i < points.size(); ++i) {
                     points_.push_back(objectToVector2D(points[i]));
                 }
                 ConstArrayAccessor1<Vector2D> pointsAcc(points_.size(),
                                                         points_.data());
                 instance.convert(pointsAcc, output.get());
             },
             R"pbdoc(
             Converts the given points to implicit surface scalar field.

             Parameters
             ----------
             - points : List of 2D vectors.
             - output : Scalar grid output.
             )pbdoc",
             py::arg("points"), py::arg("output"));
}

void addPointsToImplicit3(pybind11::module& m) {
    py::class_<PointsToImplicit3, PointsToImplicit3Ptr>(m, "PointsToImplicit3",
                                                        R"pbdoc(
        Abstract base class for 3-D points-to-implicit converters.
        )pbdoc")
        .def("convert",
             [](const PointsToImplicit3& instance, const py::list& points,
                ScalarGrid3Ptr output) {
                 std::vector<Vector3D> points_;
                 for (size_t i = 0; i < points.size(); ++i) {
                     points_.push_back(objectToVector3D(points[i]));
                 }
                 ConstArrayAccessor1<Vector3D> pointsAcc(points_.size(),
                                                         points_.data());
                 instance.convert(pointsAcc, output.get());
             },
             R"pbdoc(
             Converts the given points to implicit surface scalar field.

             Parameters
             ----------
             - points : List of 3D vectors.
             - output : Scalar grid output.
             )pbdoc",
             py::arg("points"), py::arg("output"));
}
