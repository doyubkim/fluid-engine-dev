// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "transform.h"
#include "pybind11_utils.h"

#include <jet/transform3.h>

namespace py = pybind11;
using namespace jet;

void addTransform3(pybind11::module& m) {
    py::class_<Transform3>(m, "Transform3")
        // CTOR
        .def("__init__",
             [](Transform3& instance, py::args args, py::kwargs kwargs) {
                 Transform3 tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Transform3>();
                 } else if (args.size() <= 2) {
                     tmp.setTranslation(objectToVector3D(py::object(args[0])));
                     tmp.setOrientation(objectToQuaternionD(py::object(args[1])));
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("translation")) {
                     tmp.setTranslation(objectToVector3D(py::object(kwargs["translation"])));
                 }
                 if (kwargs.contains("orientation")) {
                     tmp.setOrientation(objectToQuaternionD(py::object(kwargs["orientation"])));
                 }

                 instance = tmp;
             },
             "Constructs Transform3\n\n"
             "This method constructs 3D transform with translation and "
             "orientation.");
}
