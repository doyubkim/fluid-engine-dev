// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "ray.h"
#include "pybind11_utils.h"

#include <jet/ray2.h>
#include <jet/ray3.h>

namespace py = pybind11;
using namespace jet;

void addRay2F(pybind11::module& m) {
    py::class_<Ray2F>(m, "Ray2F")
        // CTOR
        .def("__init__",
             [](Ray2F& instance, py::args args, py::kwargs kwargs) {
                 Ray2F tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Ray2F>();
                 } else if (args.size() == 2) {
                     tmp.origin = objectToVector2F(py::object(args[0]));
                     tmp.direction = objectToVector2F(py::object(args[1]));
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("origin")) {
                     tmp.origin = objectToVector2F(py::tuple(kwargs["origin"]));
                 }
                 if (kwargs.contains("direction")) {
                     tmp.origin =
                         objectToVector2F(py::tuple(kwargs["direction"]));
                 }

                 instance = tmp;
             },
             "Constructs Ray2F\n\n"
             "This method constructs 2D float ray with origin and direction.")
        .def_readwrite("origin", &Ray2F::origin)
        .def_readwrite("direction", &Ray2F::direction);
}

void addRay2D(pybind11::module& m) {
    py::class_<Ray2D>(m, "Ray2D")
        // CTOR
        .def("__init__",
             [](Ray2D& instance, py::args args, py::kwargs kwargs) {
                 Ray2D tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Ray2D>();
                 } else if (args.size() == 2) {
                     tmp.origin = objectToVector2D(py::object(args[0]));
                     tmp.direction = objectToVector2D(py::object(args[1]));
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("origin")) {
                     tmp.origin = objectToVector2D(py::tuple(kwargs["origin"]));
                 }
                 if (kwargs.contains("direction")) {
                     tmp.origin =
                         objectToVector2D(py::tuple(kwargs["direction"]));
                 }

                 instance = tmp;
             },
             "Constructs Ray2D\n\n"
             "This method constructs 2D double ray with origin and direction.")
        .def_readwrite("origin", &Ray2D::origin)
        .def_readwrite("direction", &Ray2D::direction);
}

void addRay3F(pybind11::module& m) {
    py::class_<Ray3F>(m, "Ray3F")
        // CTOR
        .def("__init__",
             [](Ray3F& instance, py::args args, py::kwargs kwargs) {
                 Ray3F tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Ray3F>();
                 } else if (args.size() == 2) {
                     tmp.origin = objectToVector3F(py::object(args[0]));
                     tmp.direction = objectToVector3F(py::object(args[1]));
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("origin")) {
                     tmp.origin = objectToVector3F(py::tuple(kwargs["origin"]));
                 }
                 if (kwargs.contains("direction")) {
                     tmp.origin =
                         objectToVector3F(py::tuple(kwargs["direction"]));
                 }

                 instance = tmp;
             },
             "Constructs Ray3F\n\n"
             "This method constructs 3D float ray with origin and direction.")
        .def_readwrite("origin", &Ray3F::origin)
        .def_readwrite("direction", &Ray3F::direction);
}

void addRay3D(pybind11::module& m) {
    py::class_<Ray3D>(m, "Ray3D")
        // CTOR
        .def("__init__",
             [](Ray3D& instance, py::args args, py::kwargs kwargs) {
                 Ray3D tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Ray3D>();
                 } else if (args.size() == 2) {
                     tmp.origin = objectToVector3D(py::object(args[0]));
                     tmp.direction = objectToVector3D(py::object(args[1]));
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("origin")) {
                     tmp.origin = objectToVector3D(py::tuple(kwargs["origin"]));
                 }
                 if (kwargs.contains("direction")) {
                     tmp.origin =
                         objectToVector3D(py::tuple(kwargs["direction"]));
                 }

                 instance = tmp;
             },
             "Constructs Ray3D\n\n"
             "This method constructs 3D double ray with origin and direction.")
        .def_readwrite("origin", &Ray3D::origin)
        .def_readwrite("direction", &Ray3D::direction);
}
