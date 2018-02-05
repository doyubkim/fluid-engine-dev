// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "plane.h"
#include "pybind11_utils.h"

#include <jet/plane2.h>
#include <jet/plane3.h>

namespace py = pybind11;
using namespace jet;

void addPlane2(pybind11::module& m) {
    py::class_<Plane2, Plane2Ptr, Surface2>(m, "Plane2",
                                            R"pbdoc(
         2-D plane geometry.

         This class represents 2-D plane geometry which extends Surface2 by
         overriding surface-related queries.
         )pbdoc")
        // CTOR
        .def("__init__",
             [](Plane2& instance, py::object normal, py::object point,
                const Transform2& transform, bool isNormalFlipped) {
                 Vector2D normal_ = objectToVector2D(normal);
                 Vector2D point_ = objectToVector2D(point);

                 new (&instance)
                     Plane2(normal_, point_, transform, isNormalFlipped);
             },
             R"pbdoc(
             Constructs a plane that cross `point` with surface `normal`.
             )pbdoc",
             py::arg("normal"), py::arg("point"),
             py::arg("transform") = Transform2(),
             py::arg("isNormalFlipped") = false)
        .def_readwrite("normal", &Plane2::normal)
        .def_readwrite("point", &Plane2::point);
}

void addPlane3(pybind11::module& m) {
    py::class_<Plane3, Plane3Ptr, Surface3>(m, "Plane3",
                                            R"pbdoc(
         3-D plane geometry.

         This class represents 3-D plane geometry which extends Surface3 by
         overriding surface-related queries.
         )pbdoc")
        // CTOR
        .def("__init__",
             [](Plane3& instance, py::object normal, py::object point,
                const Transform3& transform, bool isNormalFlipped) {
                 Vector3D normal_ = objectToVector3D(normal);
                 Vector3D point_ = objectToVector3D(point);

                 new (&instance)
                     Plane3(normal_, point_, transform, isNormalFlipped);
             },
             R"pbdoc(
             Constructs a plane that cross `point` with surface `normal`.
             )pbdoc",
             py::arg("normal"), py::arg("point"),
             py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def_readwrite("normal", &Plane3::normal)
        .def_readwrite("point", &Plane3::point);
}
