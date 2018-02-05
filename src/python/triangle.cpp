// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "triangle.h"
#include "pybind11_utils.h"

#include <jet/triangle3.h>

namespace py = pybind11;
using namespace jet;

void addTriangle3(pybind11::module& m) {
    py::class_<Triangle3, Triangle3Ptr, Surface3>(m, "Triangle3",
                                                  R"pbdoc(
         3-D triangle geometry.

         This class represents 3-D triangle geometry which extends Surface3 by
         overriding surface-related queries.
         )pbdoc")
        // CTOR
        .def("__init__",
             [](Triangle3& instance, py::list points, py::list normals,
                py::list uvs, const Transform3& transform,
                bool isNormalFlipped) {
                 if (points.size() != 3) {
                     throw std::invalid_argument("Wrong size of point list.");
                 }
                 if (normals.size() != 3) {
                     throw std::invalid_argument("Wrong size of normal list.");
                 }
                 if (uvs.size() != 3) {
                     throw std::invalid_argument("Wrong size of uv list.");
                 }
                 std::array<Vector3D, 3> points_;
                 std::array<Vector3D, 3> normals_;
                 std::array<Vector2D, 3> uvs_;
                 for (size_t i = 0; i < 3; ++i) {
                     points_[i] = objectToVector3D(points);
                     normals_[i] = objectToVector3D(normals);
                     uvs_[i] = objectToVector2D(uvs);
                 }
                 new (&instance) Triangle3(points_, normals_, uvs_, transform,
                                           isNormalFlipped);
             },
             R"pbdoc(
             Constructs a triangle with given `points`, `normals`, and `uvs`.
             )pbdoc",
             py::arg("points"),
             py::arg("normals"),
             py::arg("uvs"),
             py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def_readwrite("points", &Triangle3::points)
        .def_readwrite("normals", &Triangle3::normals)
        .def_readwrite("uvs", &Triangle3::uvs);
}
