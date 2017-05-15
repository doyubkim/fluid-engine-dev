// Copyright (c) 2017 Doyub Kim
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
        .def("__eq__", [](const QuaternionD& instance, py::object obj) {
            QuaternionD other = objectToQuaternionD(obj);
            return instance == other;
        });
}
