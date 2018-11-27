// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "face_centered_grid.h"
#include "pybind11_utils.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <jet/face_centered_grid.h>

namespace py = pybind11;
using namespace jet;

void addFaceCenteredGrid2(py::module& m) {
    using ValueAtCenterFunc =
        Vector2D (FaceCenteredGrid2::*)(const Vector2UZ&) const;
    using DivergenceAtCellCenterFunc =
        double (FaceCenteredGrid2::*)(const Vector2UZ&) const;
    using CurlAtCellCenterFunc =
        double (FaceCenteredGrid2::*)(const Vector2UZ&) const;

    py::class_<FaceCenteredGrid2, FaceCenteredGrid2Ptr, VectorGrid2>(
        m, "FaceCenteredGrid2",
        R"pbdoc(
        2-D face-centered (a.k.a MAC or staggered) grid.

        This class implements face-centered grid which is also known as
        marker-and-cell (MAC) or staggered grid. This vector grid stores each vector
        component at face center. Thus, u and v components are not collocated.
        )pbdoc")
        .def("__init__",
             [](FaceCenteredGrid2& instance, py::args args, py::kwargs kwargs) {
                 Vector2UZ resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};
                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);
                 new (&instance)
                     FaceCenteredGrid2(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs grid.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def("set", &FaceCenteredGrid2::set,
             R"pbdoc(
             Sets the contents with the given grid.

             This method copies the given grid to this grid.

             Parameters
             ----------
             - other : Other grid to copy from.
             )pbdoc",
             py::arg("other"))
        .def("u",
             [](const FaceCenteredGrid2& instance, py::object obj) -> double {
                 return instance.u(objectToVector2UZ(obj));
             },
             R"pbdoc(
             Returns u-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("v",
             [](const FaceCenteredGrid2& instance, py::object obj) -> double {
                 return instance.v(objectToVector2UZ(obj));
             },
             R"pbdoc(
             Returns v-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("setU",
             [](FaceCenteredGrid2& instance, py::object obj, double val) {
                 instance.u(objectToVector2UZ(obj)) = val;
             },
             R"pbdoc(
             Sets u-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j).
             - val : Value to set.
             )pbdoc",
             py::arg("idx"), py::arg("val"))
        .def("setV",
             [](FaceCenteredGrid2& instance, py::object obj, double val) {
                 instance.v(objectToVector2UZ(obj)) = val;
             },
             R"pbdoc(
             Sets v-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j).
             - val : Value to set.
             )pbdoc",
             py::arg("idx"), py::arg("val"))
        .def("valueAtCellCenter",
             (ValueAtCenterFunc)&FaceCenteredGrid2::valueAtCellCenter,
             R"pbdoc(
             Returns interpolated value at cell center.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("divergenceAtCellCenter",
             (DivergenceAtCellCenterFunc)&FaceCenteredGrid2::
                 divergenceAtCellCenter,
             R"pbdoc(
             Returns divergence at cell center.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("curlAtCellCenter",
             (CurlAtCellCenterFunc)&FaceCenteredGrid2::curlAtCellCenter,
             R"pbdoc(
             Returns curl at cell center.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("vView",
             (ArrayView2<double>(FaceCenteredGrid2::*)()) &
                 FaceCenteredGrid2::vView,
             R"pbdoc(U data view.)pbdoc")
        .def("vView",
             (ArrayView2<double>(FaceCenteredGrid2::*)()) &
                 FaceCenteredGrid2::vView,
             R"pbdoc(V data view.)pbdoc")
        .def("uPosition", &FaceCenteredGrid2::uPosition,
             R"pbdoc(
            The function object that maps u data point to its actual position.
            )pbdoc")
        .def("vPosition", &FaceCenteredGrid2::vPosition,
             R"pbdoc(
            The function object that maps v data point to its actual position.
            )pbdoc")
        .def("uSize", &FaceCenteredGrid2::uSize,
             R"pbdoc(Returns data size of the u component.)pbdoc")
        .def("vSize", &FaceCenteredGrid2::vSize,
             R"pbdoc(Returns data size of the v component.)pbdoc")
        .def("uOrigin", &FaceCenteredGrid2::uOrigin,
             R"pbdoc(
             Returns u-data position for the grid point at (0, 0).

             Note that this is different from origin() since origin() returns
             the lower corner point of the bounding box.
             )pbdoc")
        .def("vOrigin", &FaceCenteredGrid2::vOrigin,
             R"pbdoc(
             Returns v-data position for the grid point at (0, 0).

             Note that this is different from origin() since origin() returns
             the lower corner point of the bounding box.
             )pbdoc")
        .def("fill",
             [](FaceCenteredGrid2& instance, py::object obj) {
                 if (py::isinstance<Vector2D>(obj)) {
                     instance.fill(obj.cast<Vector2D>());
                 } else if (py::isinstance<py::tuple>(obj)) {
                     instance.fill(objectToVector2D(obj));
                 } else if (py::isinstance<py::function>(obj)) {
                     auto func = obj.cast<py::function>();
                     instance.fill(
                         [func](const Vector2D& pt) {
                             return objectToVector2D(func(pt));
                         },
                         ExecutionPolicy::kSerial);
                 } else {
                     throw std::invalid_argument(
                         "Input type must be Vector2D or function object -> "
                         "Vector2D");
                 }
             },
             R"pbdoc(Fills the grid with given value or function.)pbdoc")
        .def("forEachUIndex",
             [](FaceCenteredGrid2& instance, py::function func) {
                 instance.forEachUIndex(func);
             },
             R"pbdoc(
             Invokes the given function func for each u-data point.

             This function invokes the given function object `func` for each u-data
             point in serial manner. The input parameters are i and j indices of a
             u-data point. The order of execution is i-first, j-last.
             )pbdoc",
             py::arg("func"))
        .def("forEachVIndex",
             [](FaceCenteredGrid2& instance, py::function func) {
                 instance.forEachVIndex(func);
             },
             R"pbdoc(
             Invokes the given function func for each v-data point.

             This function invokes the given function object `func` for each v-data
             point in serial manner. The input parameters are i and j indices of a
             v-data point. The order of execution is i-first, j-last.
             )pbdoc",
             py::arg("func"))
        .def("sample",
             [](const FaceCenteredGrid2& instance, py::object obj) {
                 return instance.sample(objectToVector2D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("divergence",
             [](const FaceCenteredGrid2& instance, py::object obj) {
                 return instance.divergence(objectToVector2D(obj));
             },
             R"pbdoc(Returns divergence at given position `x`.)pbdoc",
             py::arg("x"))
        .def("curl",
             [](const FaceCenteredGrid2& instance, py::object obj) {
                 return instance.curl(objectToVector2D(obj));
             },
             R"pbdoc(Returns curl at given position `x`.)pbdoc", py::arg("x"))
        .def("sampler", &FaceCenteredGrid2::sampler,
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

void addFaceCenteredGrid3(py::module& m) {
    using ValueAtCenterFunc =
        Vector3D (FaceCenteredGrid3::*)(const Vector3UZ&) const;
    using DivergenceAtCellCenterFunc =
        double (FaceCenteredGrid3::*)(const Vector3UZ&) const;
    using CurlAtCellCenterFunc =
        Vector3D (FaceCenteredGrid3::*)(const Vector3UZ&) const;
    using PositionFunc = GridDataPositionFunc<3> (FaceCenteredGrid3::*)() const;
    using SizeFunc = Vector3UZ (FaceCenteredGrid3::*)() const;
    using OriginFunc = Vector3D (FaceCenteredGrid3::*)() const;

    py::class_<FaceCenteredGrid3, FaceCenteredGrid3Ptr, VectorGrid3>(
        m, "FaceCenteredGrid3",
        R"pbdoc(
        3-D face-centered (a.k.a MAC or staggered) grid.

        This class implements face-centered grid which is also known as
        marker-and-cell (MAC) or staggered grid. This vector grid stores each vector
        component at face center. Thus, u, v, and w components are not collocated.
        )pbdoc")
        .def("__init__",
             [](FaceCenteredGrid3& instance, py::args args, py::kwargs kwargs) {
                 Vector3UZ resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};
                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);
                 new (&instance)
                     FaceCenteredGrid3(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs grid.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def("set", &FaceCenteredGrid3::set,
             R"pbdoc(
             Sets the contents with the given grid.

             This method copies the given grid to this grid.

             Parameters
             ----------
             - other : Other grid to copy from.
             )pbdoc",
             py::arg("other"))
        .def("u",
             [](const FaceCenteredGrid3& instance, py::object obj) -> double {
                 return instance.u(objectToVector3UZ(obj));
             },
             R"pbdoc(
             Returns u-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("v",
             [](const FaceCenteredGrid3& instance, py::object obj) -> double {
                 return instance.v(objectToVector3UZ(obj));
             },
             R"pbdoc(
             Returns v-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("w",
             [](const FaceCenteredGrid3& instance, py::object obj) -> double {
                 return instance.w(objectToVector3UZ(obj));
             },
             R"pbdoc(
             Returns v-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("setU",
             [](FaceCenteredGrid3& instance, py::object obj, double val) {
                 instance.u(objectToVector3UZ(obj)) = val;
             },
             R"pbdoc(
             Sets u-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             - val : Value to set.
             )pbdoc",
             py::arg("idx"), py::arg("val"))
        .def("setV",
             [](FaceCenteredGrid3& instance, py::object obj, double val) {
                 instance.v(objectToVector3UZ(obj)) = val;
             },
             R"pbdoc(
             Sets v-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             - val : Value to set.
             )pbdoc",
             py::arg("idx"), py::arg("val"))
        .def("setW",
             [](FaceCenteredGrid3& instance, py::object obj, double val) {
                 instance.w(objectToVector3UZ(obj)) = val;
             },
             R"pbdoc(
             Sets w-value at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             - val : Value to set.
             )pbdoc",
             py::arg("idx"), py::arg("val"))
        .def("valueAtCellCenter",
             (ValueAtCenterFunc)&FaceCenteredGrid3::valueAtCellCenter,
             R"pbdoc(
             Returns interpolated value at cell center.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("divergenceAtCellCenter",
             (DivergenceAtCellCenterFunc)&FaceCenteredGrid3::
                 divergenceAtCellCenter,
             R"pbdoc(
             Returns divergence at cell center.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("curlAtCellCenter",
             (CurlAtCellCenterFunc)&FaceCenteredGrid3::curlAtCellCenter,
             R"pbdoc(
             Returns curl at cell center.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def("vView",
             (ArrayView3<double>(FaceCenteredGrid3::*)()) &
                 FaceCenteredGrid3::vView,
             R"pbdoc(U data view.)pbdoc")
        .def("vView",
             (ArrayView3<double>(FaceCenteredGrid3::*)()) &
                 FaceCenteredGrid3::vView,
             R"pbdoc(V data view.)pbdoc")
        .def("wView",
             (ArrayView3<double>(FaceCenteredGrid3::*)()) &
                 FaceCenteredGrid3::wView,
             R"pbdoc(W data view.)pbdoc")
        .def("uPosition", &FaceCenteredGrid3::uPosition,
             R"pbdoc(
            The function object that maps u data point to its actual position.
            )pbdoc")
        .def("vPosition", &FaceCenteredGrid3::vPosition,
             R"pbdoc(
            The function object that maps v data point to its actual position.
            )pbdoc")
        .def("wPosition", (PositionFunc)&FaceCenteredGrid3::wPosition,
             R"pbdoc(
            The function object that maps w data point to its actual position.
            )pbdoc")
        .def("uSize", &FaceCenteredGrid3::uSize,
             R"pbdoc(Returns data size of the u component.)pbdoc")
        .def("vSize", &FaceCenteredGrid3::vSize,
             R"pbdoc(Returns data size of the v component.)pbdoc")
        .def("wSize", (SizeFunc)&FaceCenteredGrid3::wSize,
             R"pbdoc(Returns data size of the w component.)pbdoc")
        .def("uOrigin", &FaceCenteredGrid3::uOrigin,
             R"pbdoc(
             Returns u-data position for the grid point at (0, 0).

             Note that this is different from origin() since origin() returns
             the lower corner point of the bounding box.
             )pbdoc")
        .def("vOrigin", &FaceCenteredGrid3::vOrigin,
             R"pbdoc(
             Returns v-data position for the grid point at (0, 0).

             Note that this is different from origin() since origin() returns
             the lower corner point of the bounding box.
             )pbdoc")
        .def("wOrigin", (OriginFunc)&FaceCenteredGrid3::wOrigin,
             R"pbdoc(
             Returns w-data position for the grid point at (0, 0).

             Note that this is different from origin() since origin() returns
             the lower corner point of the bounding box.
             )pbdoc")
        .def("fill",
             [](FaceCenteredGrid3& instance, py::object obj) {
                 if (py::isinstance<Vector3D>(obj)) {
                     instance.fill(obj.cast<Vector3D>());
                 } else if (py::isinstance<py::tuple>(obj)) {
                     instance.fill(objectToVector3D(obj));
                 } else if (py::isinstance<py::function>(obj)) {
                     auto func = obj.cast<py::function>();
                     instance.fill(
                         [func](const Vector3D& pt) {
                             return objectToVector3D(func(pt));
                         },
                         ExecutionPolicy::kSerial);
                 } else {
                     throw std::invalid_argument(
                         "Input type must be Vector3D or function object -> "
                         "Vector3D");
                 }
             },
             R"pbdoc(Fills the grid with given value or function.)pbdoc")
        .def("forEachUIndex",
             [](FaceCenteredGrid3& instance, py::function func) {
                 instance.forEachUIndex(func);
             },
             R"pbdoc(
             Invokes the given function func for each u-data point.

             This function invokes the given function object `func` for each u-data
             point in serial manner. The input parameters are i, j, and k indices of a
             u-data point. The order of execution is i-first, k-last.
             )pbdoc",
             py::arg("func"))
        .def("forEachVIndex",
             [](FaceCenteredGrid3& instance, py::function func) {
                 instance.forEachVIndex(func);
             },
             R"pbdoc(
             Invokes the given function func for each v-data point.

             This function invokes the given function object `func` for each v-data
             point in serial manner. The input parameters are i, j, and k indices of a
             u-data point. The order of execution is i-first, k-last.
             )pbdoc",
             py::arg("func"))
        .def("forEachWIndex",
             [](FaceCenteredGrid3& instance, py::function func) {
                 instance.forEachWIndex(func);
             },
             R"pbdoc(
             Invokes the given function func for each w-data point.

             This function invokes the given function object `func` for each w-data
             point in serial manner. The input parameters are i, j, and k indices of a
             u-data point. The order of execution is i-first, k-last.
             )pbdoc",
             py::arg("func"))
        .def("sample",
             [](const FaceCenteredGrid3& instance, py::object obj) {
                 return instance.sample(objectToVector3D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("divergence",
             [](const FaceCenteredGrid3& instance, py::object obj) {
                 return instance.divergence(objectToVector3D(obj));
             },
             R"pbdoc(Returns divergence at given position `x`.)pbdoc",
             py::arg("x"))
        .def("curl",
             [](const FaceCenteredGrid3& instance, py::object obj) {
                 return instance.curl(objectToVector3D(obj));
             },
             R"pbdoc(Returns curl at given position `x`.)pbdoc", py::arg("x"))
        .def("sampler", &FaceCenteredGrid3::sampler,
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}
