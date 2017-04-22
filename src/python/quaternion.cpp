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
        .def(
            "__init__",
            [](QuaternionF& instance, py::args args, py::kwargs kwargs) {
                QuaternionF tmp;

                // See if we have list of parameters
                if (args.size() == 1) {
                    tmp = args[0].cast<QuaternionF>();
                } else if (args.size() == 4) {
                    tmp.w = args[0].cast<float>();
                    tmp.x = args[1].cast<float>();
                    tmp.y = args[2].cast<float>();
                    tmp.z = args[3].cast<float>();
                } else if (args.size() > 0) {
                    throw std::invalid_argument("Too few/many arguments.");
                }

                // Parse out keyword args
                if (kwargs.contains("w")) {
                    tmp.w = kwargs["w"].cast<float>();
                }
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
            "Constructs QuaternionF\n\n"
            "This method constructs float-type quaternion with w, x, y, and z.")
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
             [](QuaternionD& instance, py::args args, py::kwargs kwargs) {
                 QuaternionD tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<QuaternionD>();
                 } else if (args.size() == 4) {
                     tmp.w = args[0].cast<double>();
                     tmp.x = args[1].cast<double>();
                     tmp.y = args[2].cast<double>();
                     tmp.z = args[3].cast<double>();
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("w")) {
                     tmp.w = kwargs["w"].cast<double>();
                 }
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
             "Constructs QuaternionD\n\n"
             "This method constructs double-type quaternion with w, x, y, and "
             "z.")
        .def_readwrite("w", &QuaternionD::w)
        .def_readwrite("x", &QuaternionD::x)
        .def_readwrite("y", &QuaternionD::y)
        .def_readwrite("z", &QuaternionD::z)
        .def("__eq__", [](const QuaternionD& instance, py::object obj) {
            QuaternionD other = objectToQuaternionD(obj);
            return instance == other;
        });
}
