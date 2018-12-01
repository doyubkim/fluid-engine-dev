// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "rigid_body_collider.h"
#include "pybind11_utils.h"

#include <jet/rigid_body_collider.h>

namespace py = pybind11;
using namespace jet;

void addRigidBodyCollider2(py::module& m) {
    py::class_<RigidBodyCollider2, RigidBodyCollider2Ptr, Collider2>(
        m, "RigidBodyCollider2")
        .def(py::init([](const Surface2Ptr& surface, py::object linearVelocity,
                         double angularVelocity) {
                 return new RigidBodyCollider2(surface,
                                               objectToVector2D(linearVelocity),
                                               angularVelocity);
             }),
             R"pbdoc(
             Constructs RigidBodyCollider2

             This method constructs RigidBodyCollider2 with surface, linear
             velocity, and angular velocity.
             )pbdoc",
             py::arg("surface"), py::arg("linearVelocity") = Vector2D{},
             py::arg("angularVelocity") = 0.0)
        .def_property("linearVelocity",
                      [](const RigidBodyCollider2& instance) {
                          return instance.linearVelocity;
                      },
                      [](RigidBodyCollider2& instance, py::object obj) {
                          instance.linearVelocity = objectToVector2D(obj);
                      },
                      R"pbdoc(Linear velocity of the collider.)pbdoc")
        .def_readwrite("angularVelocity", &RigidBodyCollider2::angularVelocity,
                       R"pbdoc(Angular velocity of the collider.)pbdoc")
        .def(
            "velocityAt",
            [](const RigidBodyCollider2& instance, py::object obj) {
                return instance.velocityAt(objectToVector2D(obj));
            },
            R"pbdoc(Returns the velocity of the collider at given point.)pbdoc",
            py::arg("point"));
}

void addRigidBodyCollider3(py::module& m) {
    py::class_<RigidBodyCollider3, RigidBodyCollider3Ptr, Collider3>(
        m, "RigidBodyCollider3")
        .def(py::init([](const Surface3Ptr& surface, py::object linearVelocity,
                         py::object angularVelocity) {
                 return new RigidBodyCollider3(
                     surface, objectToVector3D(linearVelocity),
                     objectToVector3D(angularVelocity));
             }),
             R"pbdoc(
             Constructs RigidBodyCollider3

             This method constructs RigidBodyCollider3 with surface, linear
             velocity, and angular velocity.
             )pbdoc",
             py::arg("surface"), py::arg("linearVelocity") = Vector3D{},
             py::arg("angularVelocity") = Vector3D{})
        .def_property("linearVelocity",
                      [](const RigidBodyCollider3& instance) {
                          return instance.linearVelocity;
                      },
                      [](RigidBodyCollider3& instance, py::object obj) {
                          instance.linearVelocity = objectToVector3D(obj);
                      },
                      R"pbdoc(Linear velocity of the collider.)pbdoc")
        .def_property("angularVelocity",
                      [](const RigidBodyCollider3& instance) {
                          return instance.angularVelocity;
                      },
                      [](RigidBodyCollider3& instance, py::object obj) {
                          instance.angularVelocity = objectToVector3D(obj);
                      },
                      R"pbdoc(Angular velocity of the collider.)pbdoc")
        .def(
            "velocityAt",
            [](const RigidBodyCollider3& instance, py::object obj) {
                return instance.velocityAt(objectToVector3D(obj));
            },
            R"pbdoc(Returns the velocity of the collider at given point.)pbdoc",
            py::arg("point"));
}
