// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "box.h"
#include "pybind11_utils.h"

#include <jet/box2.h>
#include <jet/box3.h>

namespace py = pybind11;
using namespace jet;

void addBox2(pybind11::module& m) {
    py::class_<Box2, Box2Ptr, Surface2>(m, "Box2", R"pbdoc(
        2-D box geometry.

        This class represents 2-D box geometry which extends Surface2 by overriding
        surface-related queries. This box implementation is an axis-aligned box
        that wraps lower-level primitive type, BoundingBox2D.
        )pbdoc")
        // CTOR
        .def("__init__",
             [](Box2& instance, py::object lowerCorner, py::object upperCorner,
                const Transform2& transform, bool isNormalFlipped) {
                 new (&instance) Box2(objectToVector2D(lowerCorner),
                                      objectToVector2D(upperCorner), transform,
                                      isNormalFlipped);
             },
             R"pbdoc(
             Constructs Box2

             This method constructs Box2 with center, radius, height,
             transform, and normal direction (isNormalFlipped).
             )pbdoc",
             py::arg("lowerCorner") = Vector2D(0, 0),
             py::arg("upperCorner") = Vector2D(1, 1),
             py::arg("transform") = Transform2(),
             py::arg("isNormalFlipped") = false)
        .def_readwrite("bound", &Box2::bound);
}

void addBox3(pybind11::module& m) {
    py::class_<Box3, Box3Ptr, Surface3>(m, "Box3", R"pbdoc(
        3-D box geometry.

        This class represents 3-D box geometry which extends Surface3 by overriding
        surface-related queries. This box implementation is an axis-aligned box
        that wraps lower-level primitive type, BoundingBox3D.
        )pbdoc")
        // CTOR
        .def("__init__",
             [](Box3& instance, py::object lowerCorner, py::object upperCorner,
                const Transform3& transform, bool isNormalFlipped) {
                 new (&instance) Box3(objectToVector3D(lowerCorner),
                                      objectToVector3D(upperCorner), transform,
                                      isNormalFlipped);
             },
             R"pbdoc(
             Constructs Box3

             This method constructs Box3 with center, radius, height,
             transform, and normal direction (isNormalFlipped).
             )pbdoc",
             py::arg("lowerCorner") = Vector3D(0, 0, 0),
             py::arg("upperCorner") = Vector3D(1, 1, 1),
             py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def_readwrite("bound", &Box3::bound);
}
