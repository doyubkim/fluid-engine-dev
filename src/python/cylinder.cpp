// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cylinder.h"
#include "pybind11_utils.h"

#include <jet/cylinder3.h>

namespace py = pybind11;
using namespace jet;

void addCylinder3(pybind11::module& m) {
    py::class_<Cylinder3, Cylinder3Ptr, Surface3>(m, "Cylinder3")
        // CTOR
        .def("__init__",
             [](Cylinder3& instance, py::object center, double radius,
                double height, const Transform3& transform,
                bool isNormalFlipped) {
                 new (&instance) Cylinder3(objectToVector3D(center), radius,
                                           height, transform, isNormalFlipped);
             },
             R"pbdoc(
             Constructs Cylinder3

             This method constructs Cylinder3 with center, radius, height,
             transform, and normal direction (isNormalFlipped).
             )pbdoc",
             py::arg("center") = Vector3D(0, 0, 0), py::arg("radius") = 1.0,
             py::arg("height") = 1.0, py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def_readwrite("center", &Cylinder3::center)
        .def_readwrite("radius", &Cylinder3::radius)
        .def_readwrite("height", &Cylinder3::height);
}
