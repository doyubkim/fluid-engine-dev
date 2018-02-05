// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "transform.h"
#include "pybind11_utils.h"

#include <jet/transform2.h>
#include <jet/transform3.h>

namespace py = pybind11;
using namespace jet;

void addTransform2(pybind11::module& m) {
    py::class_<Transform2>(m, "Transform2")
        // CTOR
        .def("__init__",
             [](Transform2& instance, py::object translation,
                double orientation) {
                 Vector2D translation_ = objectToVector2D(translation);
                 new (&instance) Transform2(translation_, orientation);
             },
             R"pbdoc(
             Constructs Transform2

             This method constructs 2D transform with translation and
             orientation.
             )pbdoc",
             py::arg("translation") = Vector2D(0, 0),
             py::arg("orientation") = 0.0);
}

void addTransform3(pybind11::module& m) {
    py::class_<Transform3>(m, "Transform3")
        // CTOR
        .def("__init__",
             [](Transform3& instance, py::object translation,
                py::object orientation) {
                 Vector3D translation_ = objectToVector3D(translation);
                 QuaternionD orientation_ = objectToQuaternionD(orientation);
                 new (&instance) Transform3(translation_, orientation_);
             },
             R"pbdoc(
             Constructs Transform3

             This method constructs 3D transform with translation and
             orientation.
             )pbdoc",
             py::arg("translation") = Vector3D(0, 0, 0),
             py::arg("orientation") = QuaternionD());
}
