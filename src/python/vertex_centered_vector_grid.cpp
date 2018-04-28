// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "vertex_centered_vector_grid.h"
#include "pybind11_utils.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <jet/vertex_centered_vector_grid2.h>
#include <jet/vertex_centered_vector_grid3.h>

namespace py = pybind11;
using namespace jet;

void addVertexCenteredVectorGrid2(py::module& m) {
    py::class_<VertexCenteredVectorGrid2, VertexCenteredVectorGrid2Ptr,
               CollocatedVectorGrid2>(m, "VertexCenteredVectorGrid2",
                                      R"pbdoc(
        2-D Vertex-centered vector grid structure.

        This class represents 2-D vertex-centered vector grid which extends
        CollocatedVectorGrid2. As its name suggests, the class defines the data
        point at the center of a grid vertex. Thus, the dimension of data points are
        equal to the dimension of the vertices.
        )pbdoc")
        .def("__init__",
             [](VertexCenteredVectorGrid2& instance, py::args args,
                py::kwargs kwargs) {
                 Size2 resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};
                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);
                 new (&instance) VertexCenteredVectorGrid2(
                     resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs grid.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point at the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def_property_readonly(
            "dataSize", &VertexCenteredVectorGrid2::dataSize,
            R"pbdoc(Returns the actual data point size.)pbdoc")
        .def_property_readonly("dataOrigin",
                               &VertexCenteredVectorGrid2::dataOrigin,
                               R"pbdoc(
            Returns data position for the grid point at (0, 0).

            Note that this is different from origin() since origin() returns
            the lower corner point of the bounding box.
            )pbdoc")
        .def("fill",
             [](VertexCenteredVectorGrid2& instance, py::object obj) {
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
        .def("set", &VertexCenteredVectorGrid2::set,
             R"pbdoc(
             Sets the contents with the given `other` grid.

             Parameters
             ----------
             - other : Other grid to copy from.
             )pbdoc",
             py::arg("other"));
}

void addVertexCenteredVectorGrid3(py::module& m) {
    py::class_<VertexCenteredVectorGrid3, VertexCenteredVectorGrid3Ptr,
               CollocatedVectorGrid3>(m, "VertexCenteredVectorGrid3",
                                      R"pbdoc(
        3-D Vertex-centered vector grid structure.

        This class represents 3-D vertex-centered vector grid which extends
        CollocatedVectorGrid3. As its name suggests, the class defines the data
        point at the center of a grid vertex. Thus, the dimension of data points are
        equal to the dimension of the vertices.
        )pbdoc")
        .def("__init__",
             [](VertexCenteredVectorGrid3& instance, py::args args,
                py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};
                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);
                 new (&instance) VertexCenteredVectorGrid3(
                     resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs grid.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point at the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def_property_readonly(
            "dataSize", &VertexCenteredVectorGrid3::dataSize,
            R"pbdoc(Returns the actual data point size.)pbdoc")
        .def_property_readonly("dataOrigin",
                               &VertexCenteredVectorGrid3::dataOrigin,
                               R"pbdoc(
            Returns data position for the grid point at (0, 0, 0).

            Note that this is different from origin() since origin() returns
            the lower corner point of the bounding box.
            )pbdoc")
        .def("fill",
             [](VertexCenteredVectorGrid3& instance, py::object obj) {
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
        .def("set", &VertexCenteredVectorGrid3::set,
             R"pbdoc(
             Sets the contents with the given `other` grid.

             Parameters
             ----------
             - other : Other grid to copy from.
             )pbdoc",
             py::arg("other"));
}
