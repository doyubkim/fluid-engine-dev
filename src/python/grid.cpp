// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid.h"
#include "pybind11_utils.h"

#include <jet/grid2.h>
#include <jet/grid3.h>

namespace py = pybind11;
using namespace jet;

void addGrid2(py::module& m) {
    py::class_<Grid2, Grid2Ptr, Serializable>(m, "Grid2",
                                              R"pbdoc(
        Abstract base class for 2-D cartesian grid structure.

        This class represents 2-D cartesian grid structure. This class is an
        abstract base class and does not store any data. The class only stores the
        shape of the grid. The grid structure is axis-aligned and can have different
        grid spacing per axis.
        )pbdoc")
        .def_property_readonly("resolution", &Grid2::resolution,
                               R"pbdoc(Resolution of the grid.)pbdoc")
        .def_property_readonly("origin", &Grid2::origin,
                               R"pbdoc(Origin of the grid.)pbdoc")
        .def_property_readonly("gridSpacing", &Grid2::gridSpacing,
                               R"pbdoc(Spacing between grid points.)pbdoc")
        .def_property_readonly("boundingBox", &Grid2::boundingBox,
                               R"pbdoc(Bounding box of the entire grid.)pbdoc")
        .def_property_readonly(
            "cellCenterPosition", &Grid2::cellCenterPosition,
            R"pbdoc(Function that maps grid index to the cell-center position.)pbdoc")
        .def("forEachCellIndex",
             [](Grid2& instance, py::function func) {
                 instance.forEachCellIndex(func);
             },
             R"pbdoc(
             Invokes the given function `func` for each grid cell.

             This function invokes the given function object `func` for each grid
             cell in serial manner. The input parameters are `i` and `j` indices of a
             grid cell. The order of execution is i-first, j-last.
             )pbdoc",
             py::arg("func"))
        .def("hasSameShape",
             [](const Grid2& instance, const Grid2Ptr& other) {
                 return instance.hasSameShape(*other);
             },
             R"pbdoc(
             Returns true if resolution, grid-spacing and origin are same.
             )pbdoc")
        .def("swap",
             [](Grid2& instance, Grid2Ptr& other) {
                 return instance.swap(other.get());
             },
             R"pbdoc(
             Swaps the data with other grid.
             )pbdoc");
}

void addGrid3(py::module& m) {
    py::class_<Grid3, Grid3Ptr, Serializable>(m, "Grid3",
                                              R"pbdoc(
        Abstract base class for 3-D cartesian grid structure.

        This class represents 3-D cartesian grid structure. This class is an
        abstract base class and does not store any data. The class only stores the
        shape of the grid. The grid structure is axis-aligned and can have different
        grid spacing per axis.
        )pbdoc")
        .def_property_readonly("resolution", &Grid3::resolution,
                               R"pbdoc(Resolution of the grid.)pbdoc")
        .def_property_readonly("origin", &Grid3::origin,
                               R"pbdoc(Origin of the grid.)pbdoc")
        .def_property_readonly("gridSpacing", &Grid3::gridSpacing,
                               R"pbdoc(Spacing between grid points.)pbdoc")
        .def_property_readonly("boundingBox", &Grid3::boundingBox,
                               R"pbdoc(Bounding box of the entire grid.)pbdoc")
        .def_property_readonly(
            "cellCenterPosition", &Grid3::cellCenterPosition,
            R"pbdoc(Function that maps grid index to the cell-center position.)pbdoc")
        .def("forEachCellIndex",
             [](Grid3& instance, py::function func) {
                 instance.forEachCellIndex(func);
             },
             R"pbdoc(
             Invokes the given function `func` for each grid cell.

             This function invokes the given function object `func` for each grid
             cell in serial manner. The input parameters are `i`, `j`, and `k` indices of a
             grid cell. The order of execution is i-first, k-last.
             )pbdoc",
             py::arg("func"))
        .def("hasSameShape",
             [](const Grid3& instance, const Grid3Ptr& other) {
                 return instance.hasSameShape(*other);
             },
             R"pbdoc(
             Returns true if resolution, grid-spacing and origin are same.
             )pbdoc")
        .def("swap",
             [](Grid3& instance, Grid3Ptr& other) {
                 return instance.swap(other.get());
             },
             R"pbdoc(
             Swaps the data with other grid.
             )pbdoc");
}
