// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "scalar_grid.h"
#include "pybind11_utils.h"

#include <jet/scalar_grid.h>

namespace py = pybind11;
using namespace jet;

void addScalarGrid2(py::module& m) {
    using GradientAtDataPointFunc =
        Vector2D (ScalarGrid2::*)(const Vector2UZ&) const;
    using LaplacianAtDataPointFunc =
        double (ScalarGrid2::*)(const Vector2UZ&) const;

    py::class_<ScalarGrid2, ScalarGrid2Ptr, ScalarField2, Grid2>(
        m, "ScalarGrid2",
        R"pbdoc(Abstract base class for 2-D scalar grid structure.)pbdoc")
        .def_property_readonly("dataSize", &ScalarGrid2::dataSize,
                               R"pbdoc(
                               Returns the size of the grid data.

                               This function returns the size of the grid data which is not necessarily
                               equal to the grid resolution if the data is not stored at cell-center.
                               )pbdoc")
        .def_property_readonly("dataOrigin", &ScalarGrid2::dataOrigin,
                               R"pbdoc(
                               Returns the origin of the grid data.

                               This function returns data position for the grid point at (0, 0).
                               Note that this is different from `origin()` since `origin()` returns
                               the lower corner point of the bounding box.
                               )pbdoc")
        .def("clone", &ScalarGrid2::clone,
             R"pbdoc(Returns the copy of the grid instance.)pbdoc")
        .def("clear", &ScalarGrid2::clear,
             R"pbdoc(Clears the contents of the grid.)pbdoc")
        .def("resize",
             [](ScalarGrid2& instance, py::args args, py::kwargs kwargs) {
                 Vector2UZ resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};
                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);
                 instance.resize(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Resizes the grid using given parameters.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def("__getitem__",
             [](const ScalarGrid2& instance, py::object obj) -> double {
                 return instance(objectToVector2UZ(obj));
             },
             R"pbdoc(
             Returns the grid data at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("__setitem__",
             [](ScalarGrid2& instance, py::object obj, double val) {
                 instance(objectToVector2UZ(obj)) = val;
             },
             R"pbdoc(
            Sets the grid data at given data point.

            Parameters
            ----------
            - idx : Data point index (i, j).
            - val : Value to set.
            )pbdoc",
             py::arg("idx"), py::arg("val"))
        .def("gradientAtDataPoint",
             (GradientAtDataPointFunc)&ScalarGrid2::gradientAtDataPoint,
             R"pbdoc(
             Returns the gradient vector at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("laplacianAtDataPoint",
             (LaplacianAtDataPointFunc)&ScalarGrid2::laplacianAtDataPoint,
             R"pbdoc(
             Returns the Laplacian at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("dataView",
             (ArrayView2<double>(ScalarGrid2::*)()) & ScalarGrid2::dataView,
             R"pbdoc(The data array view.)pbdoc")
        .def("dataPosition", &ScalarGrid2::dataPosition,
             R"pbdoc(The function that maps data point to its position.)pbdoc")
        .def("fill",
             [](ScalarGrid2& instance, double value) {
                 instance.fill(value, ExecutionPolicy::kSerial);
             },
             R"pbdoc(Fills the grid with given value.)pbdoc")
        .def("fill",
             [](ScalarGrid2& instance, py::object obj) {
                 if (py::isinstance<py::function>(obj)) {
                     auto func = obj.cast<py::function>();
                     instance.fill(
                         [func](const Vector2D& pt) -> double {
                             return func(pt).cast<double>();
                         },
                         ExecutionPolicy::kSerial);
                 } else {
                     throw std::invalid_argument(
                         "Input type must be double or function object -> "
                         "double");
                 }
             },
             R"pbdoc(Fills the grid with given function.)pbdoc")
        .def("forEachDataPointIndex",
             [](ScalarGrid2& instance, py::function func) {
                 instance.forEachDataPointIndex(func);
             },
             R"pbdoc(
             Invokes the given function `func` for each data point.

             This function invokes the given function object `func` for each data
             point in serial manner. The input parameters are i and j indices of a
             data point. The order of execution is i-first, j-last.
             )pbdoc",
             py::arg("func"))
        .def("sample",
             [](const ScalarGrid2& instance, py::object obj) {
                 return instance.sample(objectToVector2D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("gradient",
             [](const ScalarGrid2& instance, py::object obj) {
                 return instance.gradient(objectToVector2D(obj));
             },
             R"pbdoc(Returns the gradient vector at given position `x`.)pbdoc",
             py::arg("x"))
        .def("laplacian",
             [](const ScalarGrid2& instance, py::object obj) {
                 return instance.laplacian(objectToVector2D(obj));
             },
             R"pbdoc(Returns the Laplacian value at given position `x`.)pbdoc",
             py::arg("x"));
}

void addScalarGrid3(py::module& m) {
    using GradientAtDataPointFunc =
        Vector3D (ScalarGrid3::*)(const Vector3UZ&) const;
    using LaplacianAtDataPointFunc =
        double (ScalarGrid3::*)(const Vector3UZ&) const;

    py::class_<ScalarGrid3, ScalarGrid3Ptr, ScalarField3, Grid3>(
        m, "ScalarGrid3",
        R"pbdoc(Abstract base class for 3-D scalar grid structure.)pbdoc")
        .def_property_readonly("dataSize", &ScalarGrid3::dataSize,
                               R"pbdoc(
                               Returns the size of the grid data.

                               This function returns the size of the grid data which is not necessarily
                               equal to the grid resolution if the data is not stored at cell-center.
                               )pbdoc")
        .def_property_readonly("dataOrigin", &ScalarGrid3::dataOrigin,
                               R"pbdoc(
                               Returns the origin of the grid data.

                               This function returns data position for the grid point at (0, 0).
                               Note that this is different from `origin()` since `origin()` returns
                               the lower corner point of the bounding box.
                               )pbdoc")
        .def("clone", &ScalarGrid3::clone,
             R"pbdoc(Returns the copy of the grid instance.)pbdoc")
        .def("clear", &ScalarGrid3::clear,
             R"pbdoc(Clears the contents of the grid.)pbdoc")
        .def("resize",
             [](ScalarGrid3& instance, py::args args, py::kwargs kwargs) {
                 Vector3UZ resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};
                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);
                 instance.resize(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Resizes the grid using given parameters.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def("__getitem__",
             [](const ScalarGrid3& instance, py::object obj) -> double {
                 return instance(objectToVector3UZ(obj));
             },
             R"pbdoc(
             Returns the grid data at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("__setitem__",
             [](ScalarGrid3& instance, py::object obj, double val) {
                 instance(objectToVector3UZ(obj)) = val;
             },
             R"pbdoc(
            Sets the grid data at given data point.

            Parameters
            ----------
            - idx : Data point index (i, j, k).
            - val : Value to set.
            )pbdoc",
             py::arg("idx"), py::arg("val"))
        .def("gradientAtDataPoint",
             (GradientAtDataPointFunc)&ScalarGrid3::gradientAtDataPoint,
             R"pbdoc(
             Returns the gradient vector at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("laplacianAtDataPoint",
             (LaplacianAtDataPointFunc)&ScalarGrid3::laplacianAtDataPoint,
             R"pbdoc(
             Returns the Laplacian at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("dataView",
             (ArrayView3<double>(ScalarGrid3::*)()) & ScalarGrid3::dataView,
             R"pbdoc(The data array view.)pbdoc")
        .def("dataPosition", &ScalarGrid3::dataPosition,
             R"pbdoc(The function that maps data point to its position.)pbdoc")
        .def("fill",
             [](ScalarGrid3& instance, double value) {
                 instance.fill(value, ExecutionPolicy::kSerial);
             },
             R"pbdoc(Fills the grid with given value.)pbdoc")
        .def("fill",
             [](ScalarGrid3& instance, py::object obj) {
                 if (py::isinstance<py::function>(obj)) {
                     auto func = obj.cast<py::function>();
                     instance.fill(
                         [func](const Vector3D& pt) -> double {
                             return func(pt).cast<double>();
                         },
                         ExecutionPolicy::kSerial);
                 } else {
                     throw std::invalid_argument(
                         "Input type must be double or function object -> "
                         "double");
                 }
             },
             R"pbdoc(Fills the grid with given function.)pbdoc")
        .def("forEachDataPointIndex",
             [](ScalarGrid3& instance, py::function func) {
                 instance.forEachDataPointIndex(func);
             },
             R"pbdoc(
             Invokes the given function `func` for each data point.

             This function invokes the given function object `func` for each data
             point in serial manner. The input parameters are i and j indices of a
             data point. The order of execution is i-first, j-last.
             )pbdoc",
             py::arg("func"))
        .def("sample",
             [](const ScalarGrid3& instance, py::object obj) {
                 return instance.sample(objectToVector3D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("gradient",
             [](const ScalarGrid3& instance, py::object obj) {
                 return instance.gradient(objectToVector3D(obj));
             },
             R"pbdoc(Returns the gradient vector at given position `x`.)pbdoc",
             py::arg("x"))
        .def("laplacian",
             [](const ScalarGrid3& instance, py::object obj) {
                 return instance.laplacian(objectToVector3D(obj));
             },
             R"pbdoc(Returns the Laplacian value at given position `x`.)pbdoc",
             py::arg("x"));
}
