// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "triangle_mesh.h"
#include "pybind11_utils.h"

#include <jet/triangle_mesh3.h>

namespace py = pybind11;
using namespace jet;

void addTriangleMesh3(pybind11::module& m) {
    py::class_<TriangleMesh3, TriangleMesh3Ptr, Surface3>(m, "TriangleMesh3",
                                                          R"pbdoc(
         3-D triangle mesh geometry.

         This class represents 3-D triangle mesh geometry which extends Surface3 by
         overriding surface-related queries. The mesh structure stores point,
         normals, and UV coordinates.
         )pbdoc")
        // CTOR
        .def("__init__",
             [](TriangleMesh3& instance, py::list points, py::list normals,
                py::list uvs, py::list pointIndices, py::list normalIndices,
                py::list uvIndices, const Transform3& transform,
                bool isNormalFlipped) {
                 TriangleMesh3::PointArray points_;
                 TriangleMesh3::NormalArray normals_;
                 TriangleMesh3::UvArray uvs_;
                 TriangleMesh3::IndexArray pointIndices_;
                 TriangleMesh3::IndexArray normalIndices_;
                 TriangleMesh3::IndexArray uvIndices_;

                 points_.resize(points.size());
                 for (size_t i = 0; i < points.size(); ++i) {
                     points_[i] = objectToVector3D(points[i]);
                 }
                 normals_.resize(normals.size());
                 for (size_t i = 0; i < normals.size(); ++i) {
                     normals_[i] = objectToVector3D(normals[i]);
                 }
                 uvs_.resize(uvs.size());
                 for (size_t i = 0; i < uvs.size(); ++i) {
                     uvs_[i] = objectToVector2D(uvs[i]);
                 }
                 pointIndices_.resize(pointIndices.size());
                 for (size_t i = 0; i < pointIndices.size(); ++i) {
                     pointIndices_[i] = objectToPoint3UI(pointIndices[i]);
                 }
                 normalIndices_.resize(normalIndices.size());
                 for (size_t i = 0; i < normalIndices.size(); ++i) {
                     normalIndices_[i] = objectToPoint3UI(normalIndices[i]);
                 }
                 uvIndices_.resize(uvIndices.size());
                 for (size_t i = 0; i < uvIndices.size(); ++i) {
                     uvIndices_[i] = objectToPoint3UI(uvIndices[i]);
                 }

                 new (&instance) TriangleMesh3(
                     points_, normals_, uvs_, pointIndices_, normalIndices_,
                     uvIndices_, transform, isNormalFlipped);
             },
             R"pbdoc(
             Constructs mesh with points, normals, uvs, and their indices.
             )pbdoc",
             py::arg("points") = py::list(), py::arg("normals") = py::list(),
             py::arg("uvs") = py::list(), py::arg("pointIndices") = py::list(),
             py::arg("normalIndices") = py::list(),
             py::arg("uvIndices") = py::list(),
             py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def("clear", &TriangleMesh3::clear,
             R"pbdoc(
             Clears all content.
             )pbdoc")
        .def("set", &TriangleMesh3::set,
             R"pbdoc(
             Copies the contents from `other` mesh.
             )pbdoc",
             py::arg("other"))
        .def("swap",
             [](TriangleMesh3& instance, const TriangleMesh3Ptr& other) {
                 instance.swap(*other);
             },
             R"pbdoc(
             Swaps the contents with `other` mesh.
             )pbdoc",
             py::arg("other"))
        .def("area", &TriangleMesh3::area,
             R"pbdoc(
             Returns area of this mesh.
             )pbdoc")
        .def("volume", &TriangleMesh3::volume,
             R"pbdoc(
             Returns volume of this mesh.
             )pbdoc")
        .def("point",
             [](const TriangleMesh3& instance, size_t i) -> Vector3D {
                 return instance.point(i);
             },
             R"pbdoc(
             Returns i-th point.
             )pbdoc",
             py::arg("i"))
        .def("setPoint",
             [](TriangleMesh3& instance, size_t i, const Vector3D& pt) {
                 instance.point(i) = pt;
             },
             R"pbdoc(
             Sets `i`-th point with `pt`.
             )pbdoc",
             py::arg("i"), py::arg("pt"))
        .def("normal",
             [](const TriangleMesh3& instance, size_t i) -> Vector3D {
                 return instance.normal(i);
             },
             R"pbdoc(
             Returns i-th normal.
             )pbdoc",
             py::arg("i"))
        .def("setNormal",
             [](TriangleMesh3& instance, size_t i, const Vector3D& n) {
                 instance.normal(i) = n;
             },
             R"pbdoc(
             Sets `i`-th normal with `pt`.
             )pbdoc",
             py::arg("i"), py::arg("n"))
        .def("pointIndex",
             [](const TriangleMesh3& instance, size_t i) -> Point3UI {
                 return instance.pointIndex(i);
             },
             R"pbdoc(
             Returns i-th pointIndex.
             )pbdoc",
             py::arg("i"))
        .def("setPointIndex",
             [](TriangleMesh3& instance, size_t i, const Point3UI& idx) {
                 instance.pointIndex(i) = idx;
             },
             R"pbdoc(
             Sets `i`-th pointIndex with `idx`.
             )pbdoc",
             py::arg("i"), py::arg("idx"))
        .def("normalIndexIndex",
             [](const TriangleMesh3& instance, size_t i) -> Point3UI {
                 return instance.normalIndex(i);
             },
             R"pbdoc(
             Returns i-th normalIndexIndex.
             )pbdoc",
             py::arg("i"))
        .def("setNormalIndexIndex",
             [](TriangleMesh3& instance, size_t i, const Point3UI& idx) {
                 instance.normalIndex(i) = idx;
             },
             R"pbdoc(
             Sets `i`-th normalIndexIndex with `idx`.
             )pbdoc",
             py::arg("i"), py::arg("idx"))
        .def("uvIndexIndex",
             [](const TriangleMesh3& instance, size_t i) -> Point3UI {
                 return instance.uvIndex(i);
             },
             R"pbdoc(
             Returns i-th uvIndexIndex.
             )pbdoc",
             py::arg("i"))
        .def("setUvIndexIndex",
             [](TriangleMesh3& instance, size_t i, const Point3UI& idx) {
                 instance.uvIndex(i) = idx;
             },
             R"pbdoc(
             Sets `i`-th uvIndexIndex with `idx`.
             )pbdoc",
             py::arg("i"), py::arg("idx"))
        .def("triangle",
             [](const TriangleMesh3& instance, size_t i) {
                 return std::make_shared<Triangle3>(instance.triangle(i));
             },
             R"pbdoc(
             Returns `i`-th triangle.
             )pbdoc",
             py::arg("i"))
        .def("numberOfPoints", &TriangleMesh3::numberOfPoints,
             R"pbdoc(
             Returns number of points.
             )pbdoc")
        .def("numberOfNormals", &TriangleMesh3::numberOfNormals,
             R"pbdoc(
             Returns number of normals.
             )pbdoc")
        .def("numberOfUvs", &TriangleMesh3::numberOfUvs,
             R"pbdoc(
             Returns number of UV coordinates.
             )pbdoc")
        .def("numberOfTriangles", &TriangleMesh3::numberOfTriangles,
             R"pbdoc(
             Returns number of triangles.
             )pbdoc")
        .def("hasNormals", &TriangleMesh3::hasNormals,
             R"pbdoc(
             Returns true if the mesh has normals.
             )pbdoc")
        .def("hasUvs", &TriangleMesh3::hasUvs,
             R"pbdoc(
             Returns true if the mesh has UV coordinates.
             )pbdoc")
        .def("addPoint",
             [](TriangleMesh3& instance, py::object obj) {
                 instance.addPoint(objectToVector3D(obj));
             },
             R"pbdoc(
             Adds a point.
             )pbdoc",
             py::arg("pt"))
        .def("addNormal",
             [](TriangleMesh3& instance, py::object obj) {
                 instance.addNormal(objectToVector3D(obj));
             },
             R"pbdoc(
             Adds a normal.
             )pbdoc",
             py::arg("n"))
        .def("addUv",
             [](TriangleMesh3& instance, py::object obj) {
                 instance.addUv(objectToVector2D(obj));
             },
             R"pbdoc(
             Adds a UV.
             )pbdoc",
             py::arg("uv"))
        .def("addPointTriangle",
             [](TriangleMesh3& instance, py::object obj) {
                 instance.addPointTriangle(objectToPoint3UI(obj));
             },
             R"pbdoc(
             Adds a triangle with points.
             )pbdoc",
             py::arg("newPointIndices"))
        .def("addPointNormalTriangle",
             [](TriangleMesh3& instance, py::object obj1, py::object obj2) {
                 instance.addPointNormalTriangle(objectToPoint3UI(obj1),
                                                 objectToPoint3UI(obj2));
             },
             R"pbdoc(
             Adds a triangle with point and normal.
             )pbdoc",
             py::arg("newPointIndices"), py::arg("newNormalIndices"))
        .def("addPointUvNormalTriangle",
             [](TriangleMesh3& instance, py::object obj1, py::object obj2,
                py::object obj3) {
                 instance.addPointUvNormalTriangle(objectToPoint3UI(obj1),
                                                   objectToPoint3UI(obj2),
                                                   objectToPoint3UI(obj3));
             },
             R"pbdoc(
             Adds a triangle with point, normal, and UV.
             )pbdoc",
             py::arg("newPointIndices"), py::arg("newUvIndices"),
             py::arg("newNormalIndices"))
        .def("addPointUvTriangle",
             [](TriangleMesh3& instance, py::object obj1, py::object obj2) {
                 instance.addPointUvTriangle(objectToPoint3UI(obj1),
                                             objectToPoint3UI(obj2));
             },
             R"pbdoc(
             Adds a triangle with point and UV.
             )pbdoc",
             py::arg("newPointIndices"), py::arg("newUvIndices"))
        .def("addTriangle", &TriangleMesh3::addTriangle,
             R"pbdoc(
             Add a triangle.
             )pbdoc",
             py::arg("tri"))
        .def("setFaceNormal", &TriangleMesh3::setFaceNormal,
             R"pbdoc(
             Sets entire normals to the face normals.
             )pbdoc")
        .def("setAngleWeightedVertexNormal",
             &TriangleMesh3::setAngleWeightedVertexNormal,
             R"pbdoc(
             Sets angle weighted vertex normal.
             )pbdoc")
        .def("scale", &TriangleMesh3::scale,
             R"pbdoc(
             Scales the mesh by given factor.
             )pbdoc",
             py::arg("factor"))
        .def("translate",
             [](TriangleMesh3& instance, py::object obj) {
                 instance.translate(objectToVector3D(obj));
             },
             R"pbdoc(
             Translates the mesh.
             )pbdoc",
             py::arg("t"))
        .def("rotate",
             [](TriangleMesh3& instance, py::object obj) {
                 instance.rotate(objectToQuaternionD(obj));
             },
             R"pbdoc(
             Rotates the mesh.
             )pbdoc",
             py::arg("rot"))
        .def("writeObj",
             [](const TriangleMesh3& instance, const std::string& filename) {
                 instance.writeObj(filename);
             },
             R"pbdoc(
             Writes the mesh in obj format to the file.
             )pbdoc",
             py::arg("filename"))
        .def("readObj",
             [](TriangleMesh3& instance, const std::string& filename) {
                 instance.readObj(filename);
             },
             R"pbdoc(
             Reads the mesh in obj format from the file.
             )pbdoc",
             py::arg("filename"));
}
