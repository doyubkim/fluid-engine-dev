// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "sphere.h"
#include "pybind11_utils.h"

#include <jet/sphere3.h>

namespace py = pybind11;
using namespace jet;

void addSphere3(pybind11::module& m) {
    py::class_<Sphere3, Sphere3Ptr, Surface3>(m, "Sphere3")
        // CTOR
        .def("__init__",
             [](Sphere3& instance, py::args args, py::kwargs kwargs) {
                 Sphere3 tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<Sphere3>();
                 } else if (args.size() >= 2) {
                     tmp.center = objectToVector3D(py::object(args[0]));
                     tmp.radius = args[1].cast<double>();
                     if (args.size() > 2) {
                         tmp.transform = args[2].cast<Transform3>();
                     }
                     if (args.size() > 3) {
                         tmp.isNormalFlipped = args[3].cast<bool>();
                     }
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("center")) {
                     tmp.center =
                         objectToVector3D(py::object(kwargs["center"]));
                 }
                 if (kwargs.contains("radius")) {
                     tmp.radius = kwargs["radius"].cast<double>();
                 }
                 if (kwargs.contains("transform")) {
                     tmp.transform = kwargs["transform"].cast<Transform3>();
                 }
                 if (kwargs.contains("isNormalFlipped")) {
                     tmp.isNormalFlipped =
                         kwargs["isNormalFlipped"].cast<bool>();
                 }

                 new (&instance) Sphere3(tmp);
             },
             "Constructs Sphere3\n\n"
             "This method constructs Sphere3 with center, radius, transform, "
             "and normal direction (isNormalFlipped).")
        .def_readwrite("center", &Sphere3::center)
        .def_readwrite("radius", &Sphere3::radius);
}