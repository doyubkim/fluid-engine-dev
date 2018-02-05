// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "sphere.h"
#include "pybind11_utils.h"

#include <jet/sphere2.h>
#include <jet/sphere3.h>

namespace py = pybind11;
using namespace jet;

void addSphere2(pybind11::module& m) {
    py::class_<Sphere2, Sphere2Ptr, Surface2>(m, "Sphere2")
        // CTOR
        .def("__init__",
             [](Sphere2& instance, py::object center, double radius,
                const Transform2& transform, bool isNormalFlipped) {
                 new (&instance) Sphere2(objectToVector2D(center), radius,
                                         transform, isNormalFlipped);
             },
             R"pbdoc(
             Constructs Sphere2.

             This method constructs Sphere2 with center, radius, transform,
             and normal direction (isNormalFlipped).
             )pbdoc",
             py::arg("center") = Vector2D{}, py::arg("radius") = 1.0,
             py::arg("transform") = Transform2(),
             py::arg("isNormalFlipped") = false)
        .def_readwrite("center", &Sphere2::center)
        .def_readwrite("radius", &Sphere2::radius);
}

void addSphere3(pybind11::module& m) {
    py::class_<Sphere3, Sphere3Ptr, Surface3>(m, "Sphere3")
        // CTOR
        .def("__init__",
             [](Sphere3& instance, py::object center, double radius,
                const Transform3& transform, bool isNormalFlipped) {
                 new (&instance) Sphere3(objectToVector3D(center), radius,
                                         transform, isNormalFlipped);
             },
             R"pbdoc(
             Constructs Sphere3.

             This method constructs Sphere3 with center, radius, transform,
             and normal direction (isNormalFlipped).
             )pbdoc",
             py::arg("center") = Vector3D{}, py::arg("radius") = 1.0,
             py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def_readwrite("center", &Sphere3::center)
        .def_readwrite("radius", &Sphere3::radius);
}
