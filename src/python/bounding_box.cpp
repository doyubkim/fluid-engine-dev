// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "bounding_box.h"
#include "pybind11_utils.h"

#include <jet/bounding_box3.h>

namespace py = pybind11;
using namespace jet;

void addBoundingBox3F(pybind11::module& m) {
    py::class_<BoundingBox3F>(m, "BoundingBox3F")
        // CTOR
        .def("__init__",
             [](BoundingBox3F& instance, py::args args, py::kwargs kwargs) {
                 BoundingBox3F tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<BoundingBox3F>();
                 } else if (args.size() == 2) {
                     tmp.lowerCorner = objectToVector3F(py::object(args[0]));
                     tmp.upperCorner = objectToVector3F(py::object(args[1]));
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("lowerCorner")) {
                     tmp.lowerCorner =
                         objectToVector3F(py::object(kwargs["lowerCorner"]));
                 }
                 if (kwargs.contains("upperCorner")) {
                     tmp.upperCorner =
                         objectToVector3F(py::object(kwargs["upperCorner"]));
                 }

                 instance = tmp;
             },
             "Constructs BoundingBox3F\n\n"
             "This method constructs 3D float bounding box with lower and "
             "upper corners.")
        .def_readwrite("lowerCorner", &BoundingBox3F::lowerCorner)
        .def_readwrite("upperCorner", &BoundingBox3F::upperCorner);
}

void addBoundingBox3D(pybind11::module& m) {
    py::class_<BoundingBox3D>(m, "BoundingBox3D")
        // CTOR
        .def("__init__",
             [](BoundingBox3D& instance, py::args args, py::kwargs kwargs) {
                 BoundingBox3D tmp;

                 // See if we have list of parameters
                 if (args.size() == 1) {
                     tmp = args[0].cast<BoundingBox3D>();
                 } else if (args.size() == 2) {
                     tmp.lowerCorner = objectToVector3D(py::object(args[0]));
                     tmp.upperCorner = objectToVector3D(py::object(args[1]));
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 // Parse out keyword args
                 if (kwargs.contains("lowerCorner")) {
                     tmp.lowerCorner =
                         objectToVector3D(py::object(kwargs["lowerCorner"]));
                 }
                 if (kwargs.contains("upperCorner")) {
                     tmp.upperCorner =
                         objectToVector3D(py::object(kwargs["upperCorner"]));
                 }

                 instance = tmp;
             },
             "Constructs BoundingBox3D\n\n"
             "This method constructs 3D double bounding box with lower and "
             "upper corners.")
        .def_readwrite("lowerCorner", &BoundingBox3D::lowerCorner)
        .def_readwrite("upperCorner", &BoundingBox3D::upperCorner);
}
