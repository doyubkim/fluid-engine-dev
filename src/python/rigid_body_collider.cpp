// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "rigid_body_collider.h"
#include "pybind11_utils.h"

#include <jet/rigid_body_collider2.h>
#include <jet/rigid_body_collider3.h>

namespace py = pybind11;
using namespace jet;

void addRigidBodyCollider2(py::module& m) {
    py::class_<RigidBodyCollider2, RigidBodyCollider2Ptr, Collider2>(
        m, "RigidBodyCollider2")
        .def(
            "__init__",
            [](RigidBodyCollider2& instance, py::args args, py::kwargs kwargs) {
                Surface2Ptr surface;
                Vector2D linearVelocity;
                double angularVelocity;

                if (args.size() >= 1 && args.size() <= 3) {
                    surface = args[0].cast<Surface2Ptr>();
                    if (args.size() > 1) {
                        linearVelocity = objectToVector2D(args[1]);
                    }
                    if (args.size() > 2) {
                        angularVelocity = args[2].cast<double>();
                    }
                } else if (args.size() > 0) {
                    throw std::invalid_argument("Too few/many arguments.");
                }

                if (kwargs.contains("surface")) {
                    surface = kwargs["surface"].cast<Surface2Ptr>();
                }
                if (kwargs.contains("linearVelocity")) {
                    linearVelocity = objectToVector2D(kwargs["linearVelocity"]);
                }
                if (kwargs.contains("angularVelocity")) {
                    angularVelocity = kwargs["angularVelocity"].cast<double>();
                }

                new (&instance) RigidBodyCollider2(surface, linearVelocity,
                                                   angularVelocity);
            },
            "Constructs RigidBodyCollider2\n\n"
            "This method constructs RigidBodyCollider2 with surface, linear "
            "velocity (optional), and angular velocity (optional).")
        .def_property("linearVelocity",
                      [](const RigidBodyCollider2& instance) {
                          return instance.linearVelocity;
                      },
                      [](RigidBodyCollider2& instance, py::object obj) {
                          instance.linearVelocity = objectToVector2D(obj);
                      })
        .def_readwrite("angularVelocity", &RigidBodyCollider2::angularVelocity)
        .def("velocityAt",
             [](const RigidBodyCollider2& instance, py::object obj) {
                 return instance.velocityAt(objectToVector2D(obj));
             });
}

void addRigidBodyCollider3(py::module& m) {
    py::class_<RigidBodyCollider3, RigidBodyCollider3Ptr, Collider3>(
        m, "RigidBodyCollider3")
        .def(
            "__init__",
            [](RigidBodyCollider3& instance, py::args args, py::kwargs kwargs) {
                Surface3Ptr surface;
                Vector3D linearVelocity;
                Vector3D angularVelocity;

                if (args.size() >= 1 && args.size() <= 3) {
                    surface = args[0].cast<Surface3Ptr>();
                    if (args.size() > 1) {
                        linearVelocity = objectToVector3D(args[1]);
                    }
                    if (args.size() > 2) {
                        angularVelocity = objectToVector3D(args[2]);
                    }
                } else if (args.size() > 0) {
                    throw std::invalid_argument("Too few/many arguments.");
                }

                if (kwargs.contains("surface")) {
                    surface = kwargs["surface"].cast<Surface3Ptr>();
                }
                if (kwargs.contains("linearVelocity")) {
                    linearVelocity = objectToVector3D(kwargs["linearVelocity"]);
                }
                if (kwargs.contains("angularVelocity")) {
                    angularVelocity =
                        objectToVector3D(kwargs["angularVelocity"]);
                }

                new (&instance) RigidBodyCollider3(surface, linearVelocity,
                                                   angularVelocity);
            },
            "Constructs RigidBodyCollider3\n\n"
            "This method constructs RigidBodyCollider3 with surface, linear "
            "velocity (optional), and angular velocity (optional).")
        .def_property("linearVelocity",
                      [](const RigidBodyCollider3& instance) {
                          return instance.linearVelocity;
                      },
                      [](RigidBodyCollider3& instance, py::object obj) {
                          instance.linearVelocity = objectToVector3D(obj);
                      })
        .def_property("angularVelocity",
                      [](const RigidBodyCollider3& instance) {
                          return instance.angularVelocity;
                      },
                      [](RigidBodyCollider3& instance, py::object obj) {
                          instance.angularVelocity = objectToVector3D(obj);
                      })
        .def("velocityAt",
             [](const RigidBodyCollider3& instance, py::object obj) {
                 return instance.velocityAt(objectToVector3D(obj));
             });
}
