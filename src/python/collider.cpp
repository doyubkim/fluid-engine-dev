// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "collider.h"
#include "pybind11_utils.h"

#include <jet/collider2.h>
#include <jet/collider3.h>

namespace py = pybind11;
using namespace jet;

void addCollider2(py::module& m) {
    py::class_<Collider2, Collider2Ptr>(m, "Collider2", R"pbdoc(
        Abstract base class for generic 2-D collider object.

        This class contains basic interfaces for colliders. Most of the
        functionalities are implemented within this class, except the member
        function Collider2::velocityAt. This class also let the subclasses to
        provide a Surface2 instance to define collider surface using
        Collider2::setSurface function.
        )pbdoc")
        .def_property("frictionCoefficient", &Collider2::frictionCoefficient,
                      &Collider2::setFrictionCoefficient, R"pbdoc(
            The friction coefficient.

            This property specifies the friction coefficient to the collider. Any
            negative inputs will be clamped to zero.
            )pbdoc")
        .def_property_readonly("surface", &Collider2::surface, R"pbdoc(
            The surface instance.
            )pbdoc")
        .def(
            "velocityAt",
            [](const Collider2& instance, py::object obj) {
                return instance.velocityAt(objectToVector2D(obj));
            },
            R"pbdoc(Returns the velocity of the collider at given point.)pbdoc",
            py::arg("point"));
}

void addCollider3(py::module& m) {
    py::class_<Collider3, Collider3Ptr>(m, "Collider3", R"pbdoc(
        Abstract base class for generic 3-D collider object.

        This class contains basic interfaces for colliders. Most of the
        functionalities are implemented within this class, except the member
        function Collider3::velocityAt. This class also let the subclasses to
        provide a Surface3 instance to define collider surface using
        Collider2::setSurface function.
        )pbdoc")
        .def_property("frictionCoefficient", &Collider3::frictionCoefficient,
                      &Collider3::setFrictionCoefficient, R"pbdoc(
            The friction coefficient.

            This property specifies the friction coefficient to the collider. Any
            negative inputs will be clamped to zero.
            )pbdoc")
        .def_property_readonly("surface", &Collider3::surface, R"pbdoc(
            The surface instance.
            )pbdoc")
        .def(
            "velocityAt",
            [](const Collider3& instance, py::object obj) {
                return instance.velocityAt(objectToVector3D(obj));
            },
            R"pbdoc(Returns the velocity of the collider at given point.)pbdoc",
            py::arg("point"));
}
