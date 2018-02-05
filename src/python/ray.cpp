// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "ray.h"
#include "pybind11_utils.h"

#include <jet/ray2.h>
#include <jet/ray3.h>

namespace py = pybind11;
using namespace jet;

void addRay2F(pybind11::module& m) {
    py::class_<Ray2F>(m, "Ray2F")
        // CTOR
        .def("__init__",
             [](Ray2F& instance, py::object origin, py::object direction) {
                 new (&instance) Ray2F(objectToVector2F(origin),
                                       objectToVector2F(direction));
             },
             R"pbdoc(
             Constructs Ray2F

             This method constructs 2D float ray with origin and direction.
             )pbdoc",
             py::arg("origin") = Vector2F{},
             py::arg("direction") = Vector2F{1, 0})
        .def_readwrite("origin", &Ray2F::origin,
                       R"pbdoc(Origin of the ray.)pbdoc")
        .def_readwrite("direction", &Ray2F::direction,
                       R"pbdoc(Direction of the ray.)pbdoc");
}

void addRay2D(pybind11::module& m) {
    py::class_<Ray2D>(m, "Ray2D")
        // CTOR
        .def("__init__",
             [](Ray2D& instance, py::object origin, py::object direction) {
                 new (&instance) Ray2D(objectToVector2D(origin),
                                       objectToVector2D(direction));
             },
             R"pbdoc(
             Constructs Ray2D

             This method constructs 2D double ray with origin and direction.
             )pbdoc",
             py::arg("origin") = Vector2D{},
             py::arg("direction") = Vector2D{1, 0})
        .def_readwrite("origin", &Ray2D::origin,
                       R"pbdoc(Origin of the ray.)pbdoc")
        .def_readwrite("direction", &Ray2D::direction,
                       R"pbdoc(Direction of the ray.)pbdoc");
}

void addRay3F(pybind11::module& m) {
    py::class_<Ray3F>(m, "Ray3F")
        // CTOR
        .def("__init__",
             [](Ray3F& instance, py::object origin, py::object direction) {
                 new (&instance) Ray3F(objectToVector3F(origin),
                                       objectToVector3F(direction));
             },
             R"pbdoc(
             Constructs Ray3F

             This method constructs 3D float ray with origin and direction.
             )pbdoc",
             py::arg("origin") = Vector3F{},
             py::arg("direction") = Vector3F{1, 0, 0})
        .def_readwrite("origin", &Ray3F::origin,
                       R"pbdoc(Origin of the ray.)pbdoc")
        .def_readwrite("direction", &Ray3F::direction,
                       R"pbdoc(Direction of the ray.)pbdoc");
}

void addRay3D(pybind11::module& m) {
    py::class_<Ray3D>(m, "Ray3D")
        // CTOR
        .def("__init__",
             [](Ray3D& instance, py::object origin, py::object direction) {
                 new (&instance) Ray3D(objectToVector3D(origin),
                                       objectToVector3D(direction));
             },
             R"pbdoc(
             Constructs Ray3D

             This method constructs 3D double ray with origin and direction.
             )pbdoc",
             py::arg("origin") = Vector3D{},
             py::arg("direction") = Vector3D{1, 0, 0})
        .def_readwrite("origin", &Ray3D::origin,
                       R"pbdoc(Origin of the ray.)pbdoc")
        .def_readwrite("direction", &Ray3D::direction,
                       R"pbdoc(Direction of the ray.)pbdoc");
}
