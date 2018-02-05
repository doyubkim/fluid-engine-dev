// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "custom_scalar_field.h"
#include "pybind11_utils.h"

#include <jet/custom_scalar_field2.h>
#include <jet/custom_scalar_field3.h>

namespace py = pybind11;
using namespace jet;

void addCustomScalarField2(py::module& m) {
    py::class_<CustomScalarField2, CustomScalarField2Ptr, ScalarField2>(
        m, "CustomScalarField2",
        R"pbdoc(2-D scalar field with custom field function.)pbdoc")
        .def("__init__",
             [](CustomScalarField2& instance, py::function func,
                py::function gradFunc, py::function lapFunc) {
                 if (!gradFunc) {
                     new (&instance) CustomScalarField2(
                         [func](const Vector2D& x) -> double {
                             return func(x).cast<double>();
                         });
                     return;
                 }
                 if (!lapFunc) {
                     new (&instance) CustomScalarField2(
                         [func](const Vector2D& x) -> double {
                             return func(x).cast<double>();
                         },
                         [gradFunc](const Vector2D& x) -> Vector2D {
                             return gradFunc(x).cast<Vector2D>();
                         });
                     return;
                 }
                 new (&instance) CustomScalarField2(
                     [func](const Vector2D& x) -> double {
                         return func(x).cast<double>();
                     },
                     [gradFunc](const Vector2D& x) -> Vector2D {
                         return gradFunc(x).cast<Vector2D>();
                     },
                     [lapFunc](const Vector2D& x) -> double {
                         return lapFunc(x).cast<double>();
                     });
             },
             R"pbdoc(
            Constructs a field with given field, gradient, and Laplacian function.
            )pbdoc",
             py::arg("func"), py::arg("gradFunc") = nullptr,
             py::arg("lapFunc") = nullptr)
        .def("sample",
             [](const CustomScalarField2& instance, py::object obj) {
                 return instance.sample(objectToVector2D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("gradient",
             [](const CustomScalarField2& instance, py::object obj) {
                 return instance.gradient(objectToVector2D(obj));
             },
             R"pbdoc(Returns gradient at given position `x`.)pbdoc",
             py::arg("x"))
        .def("laplacian",
             [](const CustomScalarField2& instance, py::object obj) {
                 return instance.laplacian(objectToVector2D(obj));
             },
             R"pbdoc(Returns Laplacian at given position `x`.)pbdoc",
             py::arg("x"))
        .def("sampler",
             [](const CustomScalarField2& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

void addCustomScalarField3(py::module& m) {
    py::class_<CustomScalarField3, CustomScalarField3Ptr, ScalarField3>(
        m, "CustomScalarField3",
        R"pbdoc(3-D scalar field with custom field function.)pbdoc")
        .def("__init__",
             [](CustomScalarField3& instance, py::function func,
                py::function gradFunc, py::function lapFunc) {
                 if (!gradFunc) {
                     new (&instance) CustomScalarField3(
                         [func](const Vector3D& x) -> double {
                             return func(x).cast<double>();
                         });
                     return;
                 }
                 if (!lapFunc) {
                     new (&instance) CustomScalarField3(
                         [func](const Vector3D& x) -> double {
                             return func(x).cast<double>();
                         },
                         [gradFunc](const Vector3D& x) -> Vector3D {
                             return gradFunc(x).cast<Vector3D>();
                         });
                     return;
                 }
                 new (&instance) CustomScalarField3(
                     [func](const Vector3D& x) -> double {
                         return func(x).cast<double>();
                     },
                     [gradFunc](const Vector3D& x) -> Vector3D {
                         return gradFunc(x).cast<Vector3D>();
                     },
                     [lapFunc](const Vector3D& x) -> double {
                         return lapFunc(x).cast<double>();
                     });
             },
             R"pbdoc(
            Constructs a field with given field, gradient, and Laplacian function.
            )pbdoc",
             py::arg("func"), py::arg("gradFunc") = nullptr,
             py::arg("lapFunc") = nullptr)
        .def("sample",
             [](const CustomScalarField3& instance, py::object obj) {
                 return instance.sample(objectToVector3D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("gradient",
             [](const CustomScalarField3& instance, py::object obj) {
                 return instance.gradient(objectToVector3D(obj));
             },
             R"pbdoc(Returns gradient at given position `x`.)pbdoc",
             py::arg("x"))
        .def("laplacian",
             [](const CustomScalarField3& instance, py::object obj) {
                 return instance.laplacian(objectToVector3D(obj));
             },
             R"pbdoc(Returns Laplacian at given position `x`.)pbdoc",
             py::arg("x"))
        .def("sampler",
             [](const CustomScalarField3& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}
