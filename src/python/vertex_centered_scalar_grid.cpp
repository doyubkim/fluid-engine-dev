// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "vertex_centered_scalar_grid.h"
#include "pybind11_utils.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <jet/vertex_centered_scalar_grid2.h>
#include <jet/vertex_centered_scalar_grid3.h>

namespace py = pybind11;
using namespace jet;

void addVertexCenteredScalarGrid2(py::module& m) {
    py::class_<VertexCenteredScalarGrid2, VertexCenteredScalarGrid2Ptr,
               ScalarGrid2>(m, "VertexCenteredScalarGrid2",
                            R"pbdoc(
        2-D Cell-centered scalar grid structure.

        This class represents 2-D cell-centered scalar grid which extends
        ScalarGrid2. As its name suggests, the class defines the data point at the
        center of a grid cell. Thus, the dimension of data points are equal to the
        dimension of the cells.
        )pbdoc")
        .def("__init__",
             [](VertexCenteredScalarGrid2& instance, py::args args,
                py::kwargs kwargs) {
                 Size2 resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};
                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);
                 new (&instance) VertexCenteredScalarGrid2(
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
            "dataSize", &VertexCenteredScalarGrid2::dataSize,
            R"pbdoc(Returns the actual data point size.)pbdoc")
        .def_property_readonly("dataOrigin",
                               &VertexCenteredScalarGrid2::dataOrigin,
                               R"pbdoc(
            Returns data position for the grid point at (0, 0).

            Note that this is different from origin() since origin() returns
            the lower corner point of the bounding box.
            )pbdoc")
        .def("set", &VertexCenteredScalarGrid2::set,
             R"pbdoc(
             Sets the contents with the given `other` grid.

             Parameters
             ----------
             - other : Other grid to copy from.
             )pbdoc",
             py::arg("other"));
}

void addVertexCenteredScalarGrid3(py::module& m) {
    py::class_<VertexCenteredScalarGrid3, VertexCenteredScalarGrid3Ptr,
               ScalarGrid3>(m, "VertexCenteredScalarGrid3",
                            R"pbdoc(
        3-D Cell-centered scalar grid structure.

        This class represents 3-D cell-centered scalar grid which extends
        ScalarGrid3. As its name suggests, the class defines the data point at the
        center of a grid cell. Thus, the dimension of data points are equal to the
        dimension of the cells.
        )pbdoc")
        .def("__init__",
             [](VertexCenteredScalarGrid3& instance, py::args args,
                py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};
                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);
                 new (&instance) VertexCenteredScalarGrid3(
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
            "dataSize", &VertexCenteredScalarGrid3::dataSize,
            R"pbdoc(Returns the actual data point size.)pbdoc")
        .def_property_readonly("dataOrigin",
                               &VertexCenteredScalarGrid3::dataOrigin,
                               R"pbdoc(
            Returns data position for the grid point at (0, 0, 0).

            Note that this is different from origin() since origin() returns
            the lower corner point of the bounding box.
            )pbdoc")
        .def("set", &VertexCenteredScalarGrid3::set,
             R"pbdoc(
             Sets the contents with the given `other` grid.

             Parameters
             ----------
             - other : Other grid to copy from.
             )pbdoc",
             py::arg("other"));
}
