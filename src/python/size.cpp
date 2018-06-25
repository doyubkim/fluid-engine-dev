// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "size.h"
#include "pybind11_utils.h"

#include <jet/matrix.h>

namespace py = pybind11;
using namespace jet;

void addVector2UZ(pybind11::module& m) {
    py::class_<Vector2UZ>(m, "Size2")
        // CTOR
        .def("__init__",
             [](Vector2UZ& instance, size_t x, size_t y) {
                 new (&instance) Vector2UZ(x, y);
             },
             R"pbdoc(
             Constructs Size2

             This method constructs 2D size with x and y.
             )pbdoc",
             py::arg("x") = 0, py::arg("y") = 0)
        .def_readwrite("x", &Vector2UZ::x)
        .def_readwrite("y", &Vector2UZ::y)
        .def("__eq__", [](const Vector2UZ& instance, py::object obj) {
            Vector2UZ other = objectToVector2UZ(obj);
            return instance == other;
        });
}

void addVector3UZ(pybind11::module& m) {
    py::class_<Vector3UZ>(m, "Size3")
        // CTOR
        .def("__init__",
             [](Vector3UZ& instance, size_t x, size_t y, size_t z) {
                 new (&instance) Vector3UZ(x, y, z);
             },
             R"pbdoc(
             Constructs Size3

             This method constructs 3D size with x, y, and z.
             )pbdoc",
             py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0)
        .def_readwrite("x", &Vector3UZ::x)
        .def_readwrite("y", &Vector3UZ::y)
        .def_readwrite("z", &Vector3UZ::z)
        .def("__eq__", [](const Vector3UZ& instance, py::object obj) {
            Vector3UZ other = objectToVector3UZ(obj);
            return instance == other;
        });
}
