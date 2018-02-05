// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "point.h"
#include "pybind11_utils.h"

#include <jet/point2.h>
#include <jet/point3.h>

namespace py = pybind11;
using namespace jet;

void addPoint2UI(pybind11::module& m) {
    py::class_<Point2UI>(m, "Point2UI")
        // CTOR
        .def("__init__", [](Point2UI& instance, size_t x,
                            size_t y) { new (&instance) Point2UI(x, y); },
             R"pbdoc(
             Constructs Point2UI

             This method constructs 2D point with x and y.
             )pbdoc", py::arg("x") = 0, py::arg("y") = 0)
        .def_readwrite("x", &Point2UI::x)
        .def_readwrite("y", &Point2UI::y)
        .def("__eq__", [](const Point2UI& instance, py::object obj) {
            Point2UI other = objectToPoint2UI(obj);
            return instance == other;
        });
}

void addPoint3UI(pybind11::module& m) {
    py::class_<Point3UI>(m, "Point3UI")
        // CTOR
        .def("__init__", [](Point3UI& instance, size_t x, size_t y,
                            size_t z) { new (&instance) Point3UI(x, y, z); },
             R"pbdoc(
             Constructs Point3UI

             This method constructs 3D point with x, y, and z.
             )pbdoc",
             py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0)
        .def_readwrite("x", &Point3UI::x)
        .def_readwrite("y", &Point3UI::y)
        .def_readwrite("z", &Point3UI::z)
        .def("__eq__", [](const Point3UI& instance, py::object obj) {
            Point3UI other = objectToPoint3UI(obj);
            return instance == other;
        });
}
