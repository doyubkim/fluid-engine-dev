// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "vector.h"
#include "pybind11_utils.h"

#include <jet/vector3.h>

namespace py = pybind11;
using namespace jet;

void addVector3F(pybind11::module& m) {
    py::class_<Vector3F>(m, "Vector3F")
        // CTOR
        .def("__init__",
             [](Vector3F& instance, py::args args, py::kwargs kwargs) {
                 Vector3F tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Vector3F>();
                 } else if (args.size() == 3) {
                     if (args.size() > 0) {
                         tmp.x = args[0].cast<float>();
                     }
                     if (args.size() > 1) {
                         tmp.y = args[1].cast<float>();
                     }
                     if (args.size() > 2) {
                         tmp.z = args[2].cast<float>();
                     }
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("x")) {
                     tmp.x = kwargs["x"].cast<float>();
                 }
                 if (kwargs.contains("y")) {
                     tmp.y = kwargs["y"].cast<float>();
                 }
                 if (kwargs.contains("z")) {
                     tmp.z = kwargs["z"].cast<float>();
                 }

                 instance = tmp;
             },
             "Constructs Vector3F\n\n"
             "This method constructs 3D float vector with x, y, and z.")
        .def_readwrite("x", &Vector3F::x)
        .def_readwrite("y", &Vector3F::y)
        .def_readwrite("z", &Vector3F::z)
        .def("__eq__", [](const Vector3F& instance, py::object obj) {
            Vector3F other = objectToVector3F(obj);
            return instance == other;
        });
    ;
}

void addVector3D(pybind11::module& m) {
    py::class_<Vector3D>(m, "Vector3D")
        // CTOR
        .def("__init__",
             [](Vector3D& instance, py::args args, py::kwargs kwargs) {
                 Vector3D tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Vector3D>();
                 } else if (args.size() == 3) {
                     if (args.size() > 0) {
                         tmp.x = args[0].cast<double>();
                     }
                     if (args.size() > 1) {
                         tmp.y = args[1].cast<double>();
                     }
                     if (args.size() > 2) {
                         tmp.z = args[2].cast<double>();
                     }
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("x")) {
                     tmp.x = kwargs["x"].cast<double>();
                 }
                 if (kwargs.contains("y")) {
                     tmp.y = kwargs["y"].cast<double>();
                 }
                 if (kwargs.contains("z")) {
                     tmp.z = kwargs["z"].cast<double>();
                 }

                 instance = tmp;
             },
             "Constructs Vector3D\n\n"
             "This method constructs 3D double vector with x, y, and z.")
        .def_readwrite("x", &Vector3D::x)
        .def_readwrite("y", &Vector3D::y)
        .def_readwrite("z", &Vector3D::z)
        .def("__eq__", [](const Vector3D& instance, py::object obj) {
            Vector3D other = objectToVector3D(obj);
            return instance == other;
        });
}
