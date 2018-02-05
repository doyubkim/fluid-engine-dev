// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "quaternion.h"
#include "pybind11_utils.h"

#include <jet/quaternion.h>

namespace py = pybind11;
using namespace jet;

void addQuaternionF(pybind11::module& m) {
    py::class_<QuaternionF>(m, "QuaternionF")
        // CTOR
        .def("__init__",
             [](QuaternionF& instance, float w, float x, float y, float z) {
                 new (&instance) QuaternionF(w, x, y, z);
             },
             R"pbdoc(
             Constructs QuaternionF.

             This method constructs float-type quaternion with w, x, y, and z.
             )pbdoc",
             py::arg("w") = 1.0f, py::arg("x") = 0.0f, py::arg("y") = 0.0f,
             py::arg("z") = 0.0f)
        .def_readwrite("w", &QuaternionF::w)
        .def_readwrite("x", &QuaternionF::x)
        .def_readwrite("y", &QuaternionF::y)
        .def_readwrite("z", &QuaternionF::z)
        .def("angle", &QuaternionF::angle)
        .def("axis", &QuaternionF::axis)
        .def("normalized", &QuaternionF::normalized)
        .def("inverse", &QuaternionF::inverse)
        .def("l2Norm", &QuaternionF::l2Norm)
        .def("setAxisAngle",
             [](QuaternionF& instance, py::object axis, float angle) {
                 instance.set(objectToVector3F(axis), angle);
             },
             R"pbdoc(
             Sets the quaternion with given rotation axis and angle.
             )pbdoc",
             py::arg("axis"), py::arg("angle"))
        .def("setFromTo",
             [](QuaternionF& instance, py::object from, py::object to) {
                 instance.set(objectToVector3F(from), objectToVector3F(to));
             },
             R"pbdoc(
             Sets the quaternion with from and to vectors.
             )pbdoc",
             py::arg("from"), py::arg("to"))
        .def("setRotationBasis",
             [](QuaternionF& instance, py::object basis0, py::object basis1,
                py::object basis2) {
                 instance.set(objectToVector3F(basis0),
                              objectToVector3F(basis1),
                              objectToVector3F(basis2));
             },
             R"pbdoc(
             Sets quaternion with three basis vectors.
             )pbdoc",
             py::arg("basis0"), py::arg("basis1"), py::arg("basis2"))
        .def("setIdentity", &QuaternionF::setIdentity)
        .def("normalize", &QuaternionF::normalize)
        .def("rotate",
             [](const QuaternionF& instance, py::object other) {
                 return instance.mul(objectToVector3F(other));
             },
             R"pbdoc(
             Returns this quaternion * other vector.
             )pbdoc",
             py::arg("other"))
        .def("dot", &QuaternionF::dot)
        .def("__mul__",
             [](const QuaternionF& instance, py::object other) {
                 return instance.mul(objectToQuaternionF(other));
             },
             R"pbdoc(
             Returns this quaternion * other quaternion.
             )pbdoc",
             py::arg("other"))
        .def("__imul__",
             [](QuaternionF& instance, py::object other) {
                 instance.imul(objectToQuaternionF(other));
             },
             R"pbdoc(
             This quaternion *= other quaternion.
             )pbdoc",
             py::arg("other"))
        .def("__setitem__", [](QuaternionF& instance, size_t i,
                               float val) { instance[i] = val; })
        .def("__getitem__", [](const QuaternionF& instance,
                               size_t i) -> float { return instance[i]; })
        .def("__eq__", [](const QuaternionF& instance, py::object obj) {
            QuaternionF other = objectToQuaternionF(obj);
            return instance == other;
        });
}

void addQuaternionD(pybind11::module& m) {
    py::class_<QuaternionD>(m, "QuaternionD")
        // CTOR
        .def("__init__",
             [](QuaternionD& instance, double w, double x, double y, double z) {
                 new (&instance) QuaternionD(w, x, y, z);
             },
             R"pbdoc(
             Constructs QuaternionD.

             This method constructs double-type quaternion with w, x, y, and z.
             )pbdoc",
             py::arg("w") = 1.0, py::arg("x") = 0.0, py::arg("y") = 0.0,
             py::arg("z") = 0.0)
        .def_readwrite("w", &QuaternionD::w)
        .def_readwrite("x", &QuaternionD::x)
        .def_readwrite("y", &QuaternionD::y)
        .def_readwrite("z", &QuaternionD::z)
        .def("angle", &QuaternionD::angle)
        .def("axis", &QuaternionD::axis)
        .def("normalized", &QuaternionD::normalized)
        .def("inverse", &QuaternionD::inverse)
        .def("l2Norm", &QuaternionD::l2Norm)
        .def("setAxisAngle",
             [](QuaternionD& instance, py::object axis, double angle) {
                 instance.set(objectToVector3D(axis), angle);
             },
             R"pbdoc(
             Sets the quaternion with given rotation axis and angle.
             )pbdoc",
             py::arg("axis"), py::arg("angle"))
        .def("setFromTo",
             [](QuaternionD& instance, py::object from, py::object to) {
                 instance.set(objectToVector3D(from), objectToVector3D(to));
             },
             R"pbdoc(
             Sets the quaternion with from and to vectors.
             )pbdoc",
             py::arg("from"), py::arg("to"))
        .def("setRotationBasis",
             [](QuaternionD& instance, py::object basis0, py::object basis1,
                py::object basis2) {
                 instance.set(objectToVector3D(basis0),
                              objectToVector3D(basis1),
                              objectToVector3D(basis2));
             },
             R"pbdoc(
             Sets quaternion with three basis vectors.
             )pbdoc",
             py::arg("basis0"), py::arg("basis1"), py::arg("basis2"))
        .def("setIdentity", &QuaternionD::setIdentity)
        .def("normalize", &QuaternionD::normalize)
        .def("rotate",
             [](const QuaternionD& instance, py::object other) {
                 return instance.mul(objectToVector3D(other));
             },
             R"pbdoc(
             Returns this quaternion * other vector.
             )pbdoc",
             py::arg("other"))
        .def("dot", &QuaternionD::dot)
        .def("__mul__",
             [](const QuaternionD& instance, py::object other) {
                 return instance.mul(objectToQuaternionD(other));
             },
             R"pbdoc(
             Returns this quaternion * other quaternion.
             )pbdoc",
             py::arg("other"))
        .def("__imul__",
             [](QuaternionD& instance, py::object other) {
                 instance.imul(objectToQuaternionD(other));
             },
             R"pbdoc(
             This quaternion *= other quaternion.
             )pbdoc",
             py::arg("other"))
        .def("__setitem__", [](QuaternionD& instance, size_t i,
                               double val) { instance[i] = val; })
        .def("__getitem__", [](const QuaternionD& instance,
                               size_t i) -> double { return instance[i]; })
        .def("__eq__", [](const QuaternionD& instance, py::object obj) {
            QuaternionD other = objectToQuaternionD(obj);
            return instance == other;
        });
}
