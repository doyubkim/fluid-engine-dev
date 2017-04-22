// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "implicit_surface3.h"
#include "pybind11_utils.h"

#include <jet/implicit_surface3.h>

namespace py = pybind11;
using namespace jet;

void addImplicitSurface3(py::module& m) {
    py::class_<ImplicitSurface3, ImplicitSurface3Ptr, Surface3>(m, "ImplicitSurface3");
//        // CTOR
//        .def("__init__",
//             [](ImplicitSurface3& instance, py::args args, py::kwargs kwargs) {
//                 Transform3 transform;
//                 bool isNormalFlipped = false;
//
//                 if (args.size() == 1) {
//                     if (py::isinstance<Transform3>(args[0])) {
//                         transform = args[0].cast<Transform3>();
//                     } else if (py::isinstance<ImplicitSurface3>(args[0])) {
//                         new (&instance) ImplicitSurface3(
//                             args[0].cast<ImplicitSurface3>(args[0]));
//                         return;
//                     }
//                 } else if (args.size() == 2) {
//                     transform = args[0].cast<Transform3>();
//                     isNormalFlipped = args[1].cast<bool>();
//                 } else if (args.size() > 0) {
//                     throw std::invalid_argument("Too few/many arguments.");
//                 }
//
//                 if (kwargs.contains("transform")) {
//                     transform = kwargs["transform"].cast<Transform3>();
//                 }
//                 if (kwargs.contains("isNormalFlipped")) {
//                     isNormalFlipped = kwargs["isNormalFlipped"].cast<bool>();
//                 }
//
//                 new (&instance) ImplicitSurface3(transform, isNormalFlipped);
//             },
//             "Constructs ImplicitSurface3\n\n"
//             "This method constructs ImplicitSurface3 with transform "
//             "(optional) and normal direction (optional).");
}
