// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "custom_vector_field.h"
#include "pybind11_utils.h"

#include <jet/custom_vector_field2.h>
#include <jet/custom_vector_field3.h>

namespace py = pybind11;
using namespace jet;

void addCustomVectorField2(py::module& m) {
    py::class_<CustomVectorField2, CustomVectorField2Ptr, VectorField2>(
        m, "CustomVectorField2",
        R"pbdoc(2-D vector field with custom field function.)pbdoc")
        .def("__init__",
             [](CustomVectorField2& instance, py::function func,
                py::function divFunc, py::function curlFunc) {
                 if (!divFunc) {
                     new (&instance) CustomVectorField2(
                         [func](const Vector2D& x) -> Vector2D {
                             return func(x).cast<Vector2D>();
                         });
                     return;
                 }
                 if (!curlFunc) {
                     new (&instance) CustomVectorField2(
                         [func](const Vector2D& x) -> Vector2D {
                             return func(x).cast<Vector2D>();
                         },
                         [divFunc](const Vector2D& x) -> double {
                             return divFunc(x).cast<double>();
                         });
                     return;
                 }
                 new (&instance) CustomVectorField2(
                     [func](const Vector2D& x) -> Vector2D {
                         return func(x).cast<Vector2D>();
                     },
                     [divFunc](const Vector2D& x) -> double {
                         return divFunc(x).cast<double>();
                     },
                     [curlFunc](const Vector2D& x) -> double {
                         return curlFunc(x).cast<double>();
                     });
             },
             R"pbdoc(
            Constructs a field with given field, gradient, and Laplacian function.
            )pbdoc",
             py::arg("func"), py::arg("divFunc") = nullptr,
             py::arg("curlFunc") = nullptr)
        .def("sample",
             [](const CustomVectorField2& instance, py::object obj) {
                 return instance.sample(objectToVector2D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("divergence",
             [](const CustomVectorField2& instance, py::object obj) {
                 return instance.divergence(objectToVector2D(obj));
             },
             R"pbdoc(Returns divergence at given position `x`.)pbdoc",
             py::arg("x"))
        .def("curl",
             [](const CustomVectorField2& instance, py::object obj) {
                 return instance.curl(objectToVector2D(obj));
             },
             R"pbdoc(Returns curl at given position `x`.)pbdoc", py::arg("x"))
        .def("sampler",
             [](const CustomVectorField2& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

void addCustomVectorField3(py::module& m) {
    py::class_<CustomVectorField3, CustomVectorField3Ptr, VectorField3>(
        m, "CustomVectorField3",
        R"pbdoc(3-D vector field with custom field function.)pbdoc")
        .def("__init__",
             [](CustomVectorField3& instance, py::function func,
                py::function divFunc, py::function curlFunc) {
                 if (!divFunc) {
                     new (&instance) CustomVectorField3(
                         [func](const Vector3D& x) -> Vector3D {
                             return func(x).cast<Vector3D>();
                         });
                     return;
                 }
                 if (!curlFunc) {
                     new (&instance) CustomVectorField3(
                         [func](const Vector3D& x) -> Vector3D {
                             return func(x).cast<Vector3D>();
                         },
                         [divFunc](const Vector3D& x) -> double {
                             return divFunc(x).cast<double>();
                         });
                     return;
                 }
                 new (&instance) CustomVectorField3(
                     [func](const Vector3D& x) -> Vector3D {
                         return func(x).cast<Vector3D>();
                     },
                     [divFunc](const Vector3D& x) -> double {
                         return divFunc(x).cast<double>();
                     },
                     [curlFunc](const Vector3D& x) -> Vector3D {
                         return curlFunc(x).cast<Vector3D>();
                     });
             },
             R"pbdoc(
            Constructs a field with given field, gradient, and Laplacian function.
            )pbdoc",
             py::arg("func"), py::arg("divFunc") = nullptr,
             py::arg("curlFunc") = nullptr)
        .def("sample",
             [](const CustomVectorField3& instance, py::object obj) {
                 return instance.sample(objectToVector3D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("divergence",
             [](const CustomVectorField3& instance, py::object obj) {
                 return instance.divergence(objectToVector3D(obj));
             },
             R"pbdoc(Returns divergence at given position `x`.)pbdoc",
             py::arg("x"))
        .def("curl",
             [](const CustomVectorField3& instance, py::object obj) {
                 return instance.curl(objectToVector3D(obj));
             },
             R"pbdoc(Returns curl at given position `x`.)pbdoc", py::arg("x"))
        .def("sampler",
             [](const CustomVectorField3& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}
