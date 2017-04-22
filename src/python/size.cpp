// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "size.h"
#include "pybind11_utils.h"

#include <jet/size3.h>

namespace py = pybind11;
using namespace jet;

void addSize2(pybind11::module& m) {
    py::class_<Size2>(m, "Size2")
        // CTOR
        .def("__init__",
             [](Size2& instance, py::args args, py::kwargs kwargs) {
                 Size2 tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Size2>();
                 } else if (args.size() == 2) {
                     if (args.size() > 0) {
                         tmp.x = args[0].cast<size_t>();
                     }
                     if (args.size() > 1) {
                         tmp.y = args[1].cast<size_t>();
                     }
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("x")) {
                     tmp.x = kwargs["x"].cast<size_t>();
                 }
                 if (kwargs.contains("y")) {
                     tmp.y = kwargs["y"].cast<size_t>();
                 }

                 instance = tmp;
             },
             "Constructs Size2\n\n"
             "This method constructs 2D size with x and y.")
        .def_readwrite("x", &Size2::x)
        .def_readwrite("y", &Size2::y)
        .def("__eq__", [](const Size2& instance, py::object obj) {
            Size2 other = objectToSize2(obj);
            return instance == other;
        });
}

void addSize3(pybind11::module& m) {
    py::class_<Size3>(m, "Size3")
        // CTOR
        .def("__init__",
             [](Size3& instance, py::args args, py::kwargs kwargs) {
                 Size3 tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Size3>();
                 } else if (args.size() == 3) {
                     if (args.size() > 0) {
                         tmp.x = args[0].cast<size_t>();
                     }
                     if (args.size() > 1) {
                         tmp.y = args[1].cast<size_t>();
                     }
                     if (args.size() > 2) {
                         tmp.z = args[2].cast<size_t>();
                     }
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("x")) {
                     tmp.x = kwargs["x"].cast<size_t>();
                 }
                 if (kwargs.contains("y")) {
                     tmp.y = kwargs["y"].cast<size_t>();
                 }
                 if (kwargs.contains("z")) {
                     tmp.z = kwargs["z"].cast<size_t>();
                 }

                 instance = tmp;
             },
             "Constructs Size3\n\n"
             "This method constructs 3D size with x, y, and z.")
        .def_readwrite("x", &Size3::x)
        .def_readwrite("y", &Size3::y)
        .def_readwrite("z", &Size3::z)
        .def("__eq__", [](const Size3& instance, py::object obj) {
            Size3 other = objectToSize3(obj);
            return instance == other;
        });
}
