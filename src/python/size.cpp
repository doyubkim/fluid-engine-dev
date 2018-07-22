// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "size.h"
#include "pybind11_utils.h"

#include <jet/size2.h>
#include <jet/size3.h>

namespace py = pybind11;
using namespace jet;

void addSize2(pybind11::module& m) {
    py::class_<Size2>(m, "Size2")
        // CTOR
        .def("__init__",
             [](Size2& instance, size_t x, size_t y) {
                 new (&instance) Size2(x, y);
             },
             R"pbdoc(
             Constructs Size2

             This method constructs 2D size with x and y.
             )pbdoc",
             py::arg("x") = 0, py::arg("y") = 0)
        .def_readwrite("x", &Size2::x)
        .def_readwrite("y", &Size2::y)
        .def("__len__", [](const Size2&) { return 2; })
        .def("__iter__",
             [](const Size2& instance) {
                 return py::make_iterator(&instance.x, &instance.y + 1);
             })
        .def("__eq__", [](const Size2& instance, py::object obj) {
            Size2 other = objectToSize2(obj);
            return instance == other;
        });
}

void addSize3(pybind11::module& m) {
    py::class_<Size3>(m, "Size3")
        // CTOR
        .def("__init__",
             [](Size3& instance, size_t x, size_t y, size_t z) {
                 new (&instance) Size3(x, y, z);
             },
             R"pbdoc(
             Constructs Size3

             This method constructs 3D size with x, y, and z.
             )pbdoc",
             py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0)
        .def_readwrite("x", &Size3::x)
        .def_readwrite("y", &Size3::y)
        .def_readwrite("z", &Size3::z)
        .def("__len__", [](const Size3&) { return 3; })
        .def("__iter__",
             [](const Size3& instance) {
                 return py::make_iterator(&instance.x, &instance.z + 1);
             })
        .def("__eq__", [](const Size3& instance, py::object obj) {
            Size3 other = objectToSize3(obj);
            return instance == other;
        });
}
