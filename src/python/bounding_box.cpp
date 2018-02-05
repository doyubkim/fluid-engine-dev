// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "bounding_box.h"
#include "pybind11_utils.h"

#include <jet/bounding_box2.h>
#include <jet/bounding_box3.h>

namespace py = pybind11;
using namespace jet;

void addBoundingBox2F(pybind11::module& m) {
    py::class_<BoundingBoxRayIntersection2F>(m, "BoundingBoxRayIntersection2F",
                                             R"pbdoc(
        2-D box-ray intersection result (32-bit float).)pbdoc")
        .def_readwrite("isIntersecting",
                       &BoundingBoxRayIntersection2F::isIntersecting,
                       R"pbdoc(True if the box and ray intersects.)pbdoc")
        .def_readwrite("tNear", &BoundingBoxRayIntersection2F::tNear,
                       R"pbdoc(Distance to the first intersection point.)pbdoc")
        .def_readwrite("tFar", &BoundingBoxRayIntersection2F::tFar,
                       R"pbdoc(
                       Distance to the second (and the last) intersection point.
                       )pbdoc");

    py::class_<BoundingBox2F>(m, "BoundingBox2F", R"pbdoc(
        2-D axis-aligned bounding box class (32-bit float).
        )pbdoc")
        .def("__init__",
             [](BoundingBox2F& instance, py::args args) {
                 if (args.size() == 2) {
                     new (&instance) BoundingBox2F(objectToVector2F(args[0]),
                                                   objectToVector2F(args[1]));
                 } else if (args.size() == 0) {
                     new (&instance) BoundingBox2F();
                 }
             },
             R"pbdoc(
             Constructs BoundingBox2F

             This method constructs 2-D bounding box with lower and upper
             corners with 32-bit precision.

             Parameters
             ----------
             - `*args` : Lower and upper corner of the bounding box. If empty,
                         empty box will be created. Must be size of 0 or 2.
             )pbdoc")
        .def_readwrite("lowerCorner", &BoundingBox2F::lowerCorner, R"pbdoc(
             Lower corner of the bounding box.)pbdoc")
        .def_readwrite("upperCorner", &BoundingBox2F::upperCorner, R"pbdoc(
             Upper corner of the bounding box.)pbdoc")
        .def_property_readonly("width", &BoundingBox2F::width, R"pbdoc(
             Width of the box.)pbdoc")
        .def_property_readonly("height", &BoundingBox2F::height,
                               R"pbdoc(Height of the box.)pbdoc")
        .def("length", &BoundingBox2F::length, R"pbdoc(
             Returns length of the box in given axis.

             Parameters
             ----------
             - axis : 0 or 1.
             )pbdoc", py::arg("axis"))
        .def("overlaps", &BoundingBox2F::overlaps, R"pbdoc(
             Returns true of this box and other box overlaps.

             Parameters
             ----------
             - other : Other bounding box to test with.
             )pbdoc", py::arg("other"))
        .def("contains",
             [](const BoundingBox2F& instance, py::object point) {
                 return instance.contains(objectToVector2F(point));
             },
             R"pbdoc(
             Returns true if the input vector is inside of this box.

             Parameters
             ----------
             - point : Point to test.
             )pbdoc",
             py::arg("point"))
        .def("intersects", &BoundingBox2F::intersects,
             R"pbdoc(
             Returns true if the input ray is intersecting with this box.

             Parameters
             ----------
             - ray : Ray to test.
             )pbdoc",
             py::arg("ray"))
        .def("closestIntersection", &BoundingBox2F::closestIntersection,
             R"pbdoc(
             Returns closest intersection for given ray.

             Returns intersection.isIntersecting = true if the input ray is
             intersecting with this box. If interesects, intersection.tNear is
             assigned with distant to the closest intersecting point, and
             intersection.tFar with furthest.

             Parameters
             ----------
             - ray : Ray to test.
             )pbdoc",
             py::arg("ray"))
        .def("midPoint", &BoundingBox2F::midPoint, R"pbdoc(
             Returns the mid-point of this box.)pbdoc")
        .def("diagonalLength", &BoundingBox2F::diagonalLength, R"pbdoc(
             Returns diagonal length of this box.)pbdoc")
        .def("diagonalLengthSquared", &BoundingBox2F::diagonalLengthSquared,
             R"pbdoc(
             Returns squared diagonal length of this box.
             )pbdoc")
        .def("reset", &BoundingBox2F::reset, R"pbdoc(
             Resets this box to initial state (min=infinite, max=-infinite).
             )pbdoc")
        .def("merge",
             [](BoundingBox2F& instance, py::object other) {
                 if (py::isinstance<py::tuple>(other) ||
                     py::isinstance<Vector2F>(other)) {
                     instance.merge(objectToVector2F(other));
                 } else if (py::isinstance<BoundingBox2F>(other)) {
                     instance.merge(other.cast<BoundingBox2F>());
                 } else {
                     throw std::invalid_argument(
                         "Incompatible argument type for merge.");
                 }
             },
             R"pbdoc(
             Merges this and other point or box.

             Parameters
             ----------
             - other : Other point or bounding box to test with.
             )pbdoc",
             py::arg("other"))
        .def("expand", &BoundingBox2F::expand, R"pbdoc(
             Expands this box by given delta to all direction.

             If the width of the box was x, expand(y) will result a box with
             x+y+y width.

             Parameters
             ----------
             - delta : Amount to expand.
             )pbdoc", py::arg("delta"))
        .def("corner", &BoundingBox2F::corner, R"pbdoc(
             Returns corner position. Index starts from x-first order.

             Parameters
             ----------
             - idx : Index of the corner.
             )pbdoc", py::arg("idx"))
        .def("clamp", &BoundingBox2F::clamp, R"pbdoc(
             Returns the clamped point.

             Parameters
             ----------
             - point : Point to clamp.
             )pbdoc", py::arg("point"))
        .def("isEmpty", &BoundingBox2F::isEmpty, R"pbdoc(
             Returns true if the box is empty.
             )pbdoc");
}

void addBoundingBox2D(pybind11::module& m) {
    py::class_<BoundingBoxRayIntersection2D>(m, "BoundingBoxRayIntersection2D",
                                             R"pbdoc(
        2-D box-ray intersection result (64-bit float).)pbdoc")
        .def_readwrite("isIntersecting",
                       &BoundingBoxRayIntersection2D::isIntersecting,
                       R"pbdoc(True if the box and ray intersects.)pbdoc")
        .def_readwrite("tNear", &BoundingBoxRayIntersection2D::tNear,
                       R"pbdoc(Distance to the first intersection point.)pbdoc")
        .def_readwrite("tFar", &BoundingBoxRayIntersection2D::tFar,
                       R"pbdoc(
                       Distance to the second (and the last) intersection point.
                       )pbdoc");

    py::class_<BoundingBox2D>(m, "BoundingBox2D", R"pbdoc(
        2-D axis-aligned bounding box class (64-bit float).
        )pbdoc")
        .def("__init__",
             [](BoundingBox2D& instance, py::args args) {
                 if (args.size() == 2) {
                     new (&instance) BoundingBox2D(objectToVector2D(args[0]),
                                                   objectToVector2D(args[1]));
                 } else if (args.size() == 0) {
                     new (&instance) BoundingBox2D();
                 }
             },
             R"pbdoc(
             Constructs BoundingBox2D

             This method constructs 2-D bounding box with lower and upper
             corners with 64-bit precision.

             Parameters
             ----------
             - `*args` : Lower and upper corner of the bounding box. If empty,
                         empty box will be created. Must be size of 0 or 2.
             )pbdoc")
        .def_readwrite("lowerCorner", &BoundingBox2D::lowerCorner, R"pbdoc(
             Lower corner of the bounding box.)pbdoc")
        .def_readwrite("upperCorner", &BoundingBox2D::upperCorner, R"pbdoc(
             Upper corner of the bounding box.)pbdoc")
        .def_property_readonly("width", &BoundingBox2D::width, R"pbdoc(
             Width of the box.)pbdoc")
        .def_property_readonly("height", &BoundingBox2D::height,
                               R"pbdoc(Height of the box.)pbdoc")
        .def("length", &BoundingBox2D::length, R"pbdoc(
             Returns length of the box in given axis.

             Parameters
             ----------
             - axis : 0 or 1.
             )pbdoc", py::arg("axis"))
        .def("overlaps", &BoundingBox2D::overlaps, R"pbdoc(
             Returns true of this box and other box overlaps.

             Parameters
             ----------
             - other : Other bounding box to test with.
             )pbdoc", py::arg("other"))
        .def("contains",
             [](const BoundingBox2D& instance, py::object point) {
                 return instance.contains(objectToVector2D(point));
             },
             R"pbdoc(
             Returns true if the input vector is inside of this box.

             Parameters
             ----------
             - point : Point to test.
             )pbdoc",
             py::arg("point"))
        .def("intersects", &BoundingBox2D::intersects,
             R"pbdoc(
             Returns true if the input ray is intersecting with this box.

             Parameters
             ----------
             - ray : Ray to test.
             )pbdoc",
             py::arg("ray"))
        .def("closestIntersection", &BoundingBox2D::closestIntersection,
             R"pbdoc(
             Returns closest intersection for given ray.

             Returns intersection.isIntersecting = true if the input ray is
             intersecting with this box. If interesects, intersection.tNear is
             assigned with distant to the closest intersecting point, and
             intersection.tFar with furthest.

             Parameters
             ----------
             - ray : Ray to test.
             )pbdoc",
             py::arg("ray"))
        .def("midPoint", &BoundingBox2D::midPoint, R"pbdoc(
             Returns the mid-point of this box.)pbdoc")
        .def("diagonalLength", &BoundingBox2D::diagonalLength, R"pbdoc(
             Returns diagonal length of this box.)pbdoc")
        .def("diagonalLengthSquared", &BoundingBox2D::diagonalLengthSquared,
             R"pbdoc(
             Returns squared diagonal length of this box.
             )pbdoc")
        .def("reset", &BoundingBox2D::reset, R"pbdoc(
             Resets this box to initial state (min=infinite, max=-infinite).
             )pbdoc")
        .def("merge",
             [](BoundingBox2D& instance, py::object other) {
                 if (py::isinstance<py::tuple>(other) ||
                     py::isinstance<Vector2D>(other)) {
                     instance.merge(objectToVector2D(other));
                 } else if (py::isinstance<BoundingBox2D>(other)) {
                     instance.merge(other.cast<BoundingBox2D>());
                 } else {
                     throw std::invalid_argument(
                         "Incompatible argument type for merge.");
                 }
             },
             R"pbdoc(
             Merges this and other point or box.

             Parameters
             ----------
             - other : Other point or bounding box to test with.
             )pbdoc",
             py::arg("other"))
        .def("expand", &BoundingBox2D::expand, R"pbdoc(
             Expands this box by given delta to all direction.

             If the width of the box was x, expand(y) will result a box with
             x+y+y width.

             Parameters
             ----------
             - delta : Amount to expand.
             )pbdoc", py::arg("delta"))
        .def("corner", &BoundingBox2D::corner, R"pbdoc(
             Returns corner position. Index starts from x-first order.

             Parameters
             ----------
             - idx : Index of the corner.
             )pbdoc", py::arg("idx"))
        .def("clamp", &BoundingBox2D::clamp, R"pbdoc(
             Returns the clamped point.

             Parameters
             ----------
             - point : Point to clamp.
             )pbdoc", py::arg("point"))
        .def("isEmpty", &BoundingBox2D::isEmpty, R"pbdoc(
             Returns true if the box is empty.
             )pbdoc");
}

void addBoundingBox3F(pybind11::module& m) {
    py::class_<BoundingBoxRayIntersection3F>(m, "BoundingBoxRayIntersection3F",
                                             R"pbdoc(
        3-D box-ray intersection result (32-bit float).)pbdoc")
        .def_readwrite("isIntersecting",
                       &BoundingBoxRayIntersection3F::isIntersecting,
                       R"pbdoc(True if the box and ray intersects.)pbdoc")
        .def_readwrite("tNear", &BoundingBoxRayIntersection3F::tNear,
                       R"pbdoc(Distance to the first intersection point.)pbdoc")
        .def_readwrite("tFar", &BoundingBoxRayIntersection3F::tFar,
                       R"pbdoc(
                       Distance to the second (and the last) intersection point.
                       )pbdoc");

    py::class_<BoundingBox3F>(m, "BoundingBox3F", R"pbdoc(
        3-D axis-aligned bounding box class (32-bit float).
        )pbdoc")
        .def("__init__",
             [](BoundingBox3F& instance, py::args args) {
                 if (args.size() == 2) {
                     new (&instance) BoundingBox3F(objectToVector3F(args[0]),
                                                   objectToVector3F(args[1]));
                 } else if (args.size() == 0) {
                     new (&instance) BoundingBox3F();
                 }
             },
             R"pbdoc(
             Constructs BoundingBox3F

             This method constructs 3-D bounding box with lower and upper
             corners with 32-bit precision.

             Parameters
             ----------
             - `*args` : Lower and upper corner of the bounding box. If empty,
                         empty box will be created. Must be size of 0 or 2.
             )pbdoc")
        .def_readwrite("lowerCorner", &BoundingBox3F::lowerCorner, R"pbdoc(
             Lower corner of the bounding box.)pbdoc")
        .def_readwrite("upperCorner", &BoundingBox3F::upperCorner, R"pbdoc(
             Upper corner of the bounding box.)pbdoc")
        .def_property_readonly("width", &BoundingBox3F::width, R"pbdoc(
             Width of the box.)pbdoc")
        .def_property_readonly("height", &BoundingBox3F::height,
                               R"pbdoc(Height of the box.)pbdoc")
        .def_property_readonly("depth", &BoundingBox3F::depth, R"pbdoc(
             Depth of the box.)pbdoc")
        .def("length", &BoundingBox3F::length, R"pbdoc(
             Returns length of the box in given axis.

             Parameters
             ----------
             - axis : 0, 1, or 2.
             )pbdoc", py::arg("axis"))
        .def("overlaps", &BoundingBox3F::overlaps, R"pbdoc(
             Returns true of this box and other box overlaps.

             Parameters
             ----------
             - other : Other bounding box to test with.
             )pbdoc", py::arg("other"))
        .def("contains",
             [](const BoundingBox3F& instance, py::object point) {
                 return instance.contains(objectToVector3F(point));
             },
             R"pbdoc(
             Returns true if the input vector is inside of this box.

             Parameters
             ----------
             - point : Point to test.
             )pbdoc",
             py::arg("point"))
        .def("intersects", &BoundingBox3F::intersects,
             R"pbdoc(
             Returns true if the input ray is intersecting with this box.

             Parameters
             ----------
             - ray : Ray to test.
             )pbdoc",
             py::arg("ray"))
        .def("closestIntersection", &BoundingBox3F::closestIntersection,
             R"pbdoc(
             Returns closest intersection for given ray.

             Returns intersection.isIntersecting = true if the input ray is
             intersecting with this box. If interesects, intersection.tNear is
             assigned with distant to the closest intersecting point, and
             intersection.tFar with furthest.

             Parameters
             ----------
             - ray : Ray to test.
             )pbdoc",
             py::arg("ray"))
        .def("midPoint", &BoundingBox3F::midPoint, R"pbdoc(
             Returns the mid-point of this box.)pbdoc")
        .def("diagonalLength", &BoundingBox3F::diagonalLength, R"pbdoc(
             Returns diagonal length of this box.)pbdoc")
        .def("diagonalLengthSquared", &BoundingBox3F::diagonalLengthSquared,
             R"pbdoc(
             Returns squared diagonal length of this box.
             )pbdoc")
        .def("reset", &BoundingBox3F::reset, R"pbdoc(
             Resets this box to initial state (min=infinite, max=-infinite).
             )pbdoc")
        .def("merge",
             [](BoundingBox3F& instance, py::object other) {
                 if (py::isinstance<py::tuple>(other) ||
                     py::isinstance<Vector3F>(other)) {
                     instance.merge(objectToVector3F(other));
                 } else if (py::isinstance<BoundingBox3F>(other)) {
                     instance.merge(other.cast<BoundingBox3F>());
                 } else {
                     throw std::invalid_argument(
                         "Incompatible argument type for merge.");
                 }
             },
             R"pbdoc(
             Merges this and other point or box.

             Parameters
             ----------
             - other : Other point or bounding box to test with.
             )pbdoc",
             py::arg("other"))
        .def("expand", &BoundingBox3F::expand, R"pbdoc(
             Expands this box by given delta to all direction.

             If the width of the box was x, expand(y) will result a box with
             x+y+y width.

             Parameters
             ----------
             - delta : Amount to expand.
             )pbdoc", py::arg("delta"))
        .def("corner", &BoundingBox3F::corner, R"pbdoc(
             Returns corner position. Index starts from x-first order.

             Parameters
             ----------
             - idx : Index of the corner.
             )pbdoc", py::arg("idx"))
        .def("clamp", &BoundingBox3F::clamp, R"pbdoc(
             Returns the clamped point.

             Parameters
             ----------
             - point : Point to clamp.
             )pbdoc", py::arg("point"))
        .def("isEmpty", &BoundingBox3F::isEmpty, R"pbdoc(
             Returns true if the box is empty.
             )pbdoc");
}

void addBoundingBox3D(pybind11::module& m) {
    py::class_<BoundingBoxRayIntersection3D>(m, "BoundingBoxRayIntersection3D",
                                             R"pbdoc(
        3-D box-ray intersection result (64-bit float).)pbdoc")
        .def_readwrite("isIntersecting",
                       &BoundingBoxRayIntersection3D::isIntersecting,
                       R"pbdoc(True if the box and ray intersects.)pbdoc")
        .def_readwrite("tNear", &BoundingBoxRayIntersection3D::tNear,
                       R"pbdoc(Distance to the first intersection point.)pbdoc")
        .def_readwrite("tFar", &BoundingBoxRayIntersection3D::tFar,
                       R"pbdoc(
                       Distance to the second (and the last) intersection point.
                       )pbdoc");

    py::class_<BoundingBox3D>(m, "BoundingBox3D", R"pbdoc(
        3-D axis-aligned bounding box class (64-bit float).
        )pbdoc")
        .def("__init__",
             [](BoundingBox3D& instance, py::args args) {
                 if (args.size() == 2) {
                     new (&instance) BoundingBox3D(objectToVector3D(args[0]),
                                                   objectToVector3D(args[1]));
                 } else if (args.size() == 0) {
                     new (&instance) BoundingBox3D();
                 }
             },
             R"pbdoc(
             Constructs BoundingBox3D

             This method constructs 3-D bounding box with lower and upper
             corners with 64-bit precision.

             Parameters
             ----------
             - `*args` : Lower and upper corner of the bounding box. If empty,
                         empty box will be created. Must be size of 0 or 2.
             )pbdoc")
        .def_readwrite("lowerCorner", &BoundingBox3D::lowerCorner, R"pbdoc(
             Lower corner of the bounding box.)pbdoc")
        .def_readwrite("upperCorner", &BoundingBox3D::upperCorner, R"pbdoc(
             Upper corner of the bounding box.)pbdoc")
        .def_property_readonly("width", &BoundingBox3D::width, R"pbdoc(
             Width of the box.)pbdoc")
        .def_property_readonly("height", &BoundingBox3D::height,
                               R"pbdoc(Height of the box.)pbdoc")
        .def_property_readonly("depth", &BoundingBox3D::depth, R"pbdoc(
             Depth of the box.)pbdoc")
        .def("length", &BoundingBox3D::length, R"pbdoc(
             Returns length of the box in given axis.

             Parameters
             ----------
             - axis : 0, 1, or 2.
             )pbdoc", py::arg("axis"))
        .def("overlaps", &BoundingBox3D::overlaps, R"pbdoc(
             Returns true of this box and other box overlaps.

             Parameters
             ----------
             - other : Other bounding box to test with.
             )pbdoc", py::arg("other"))
        .def("contains",
             [](const BoundingBox3D& instance, py::object point) {
                 return instance.contains(objectToVector3D(point));
             },
             R"pbdoc(
             Returns true if the input vector is inside of this box.

             Parameters
             ----------
             - point : Point to test.
             )pbdoc",
             py::arg("point"))
        .def("intersects", &BoundingBox3D::intersects,
             R"pbdoc(
             Returns true if the input ray is intersecting with this box.

             Parameters
             ----------
             - ray : Ray to test.
             )pbdoc",
             py::arg("ray"))
        .def("closestIntersection", &BoundingBox3D::closestIntersection,
             R"pbdoc(
             Returns closest intersection for given ray.

             Returns intersection.isIntersecting = true if the input ray is
             intersecting with this box. If interesects, intersection.tNear is
             assigned with distant to the closest intersecting point, and
             intersection.tFar with furthest.

             Parameters
             ----------
             - ray : Ray to test.
             )pbdoc",
             py::arg("ray"))
        .def("midPoint", &BoundingBox3D::midPoint, R"pbdoc(
             Returns the mid-point of this box.)pbdoc")
        .def("diagonalLength", &BoundingBox3D::diagonalLength, R"pbdoc(
             Returns diagonal length of this box.)pbdoc")
        .def("diagonalLengthSquared", &BoundingBox3D::diagonalLengthSquared,
             R"pbdoc(
             Returns squared diagonal length of this box.
             )pbdoc")
        .def("reset", &BoundingBox3D::reset, R"pbdoc(
             Resets this box to initial state (min=infinite, max=-infinite).
             )pbdoc")
        .def("merge",
             [](BoundingBox3D& instance, py::object other) {
                 if (py::isinstance<py::tuple>(other) ||
                     py::isinstance<Vector3D>(other)) {
                     instance.merge(objectToVector3D(other));
                 } else if (py::isinstance<BoundingBox3D>(other)) {
                     instance.merge(other.cast<BoundingBox3D>());
                 } else {
                     throw std::invalid_argument(
                         "Incompatible argument type for merge.");
                 }
             },
             R"pbdoc(
             Merges this and other point or box.

             Parameters
             ----------
             - other : Other point or bounding box to test with.
             )pbdoc",
             py::arg("other"))
        .def("expand", &BoundingBox3D::expand, R"pbdoc(
             Expands this box by given delta to all direction.

             If the width of the box was x, expand(y) will result a box with
             x+y+y width.

             Parameters
             ----------
             - delta : Amount to expand.
             )pbdoc", py::arg("delta"))
        .def("corner", &BoundingBox3D::corner, R"pbdoc(
             Returns corner position. Index starts from x-first order.

             Parameters
             ----------
             - idx : Index of the corner.
             )pbdoc", py::arg("idx"))
        .def("clamp", &BoundingBox3D::clamp, R"pbdoc(
             Returns the clamped point.

             Parameters
             ----------
             - point : Point to clamp.
             )pbdoc", py::arg("point"))
        .def("isEmpty", &BoundingBox3D::isEmpty, R"pbdoc(
             Returns true if the box is empty.
             )pbdoc");
}