// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_system_data.h"
#include "pybind11_utils.h"

#include <jet/grid_system_data2.h>
#include <jet/grid_system_data3.h>

namespace py = pybind11;
using namespace jet;

void addGridSystemData2(py::module& m) {
    py::class_<GridSystemData2, GridSystemData2Ptr, Serializable>(
        m, "GridSystemData2",
        R"pbdoc(
        2-D grid system data.

        This class is the key data structure for storing grid system data. To
        represent a grid system for fluid simulation, velocity field is defined as a
        face-centered (MAC) grid by default. It can also have additional scalar or
        vector attributes by adding extra data layer.
        )pbdoc")
        .def(py::init<>())
        .def_property_readonly("resolution", &GridSystemData2::resolution,
                               R"pbdoc(Resolution of the grid.)pbdoc")
        .def_property_readonly("origin", &GridSystemData2::origin,
                               R"pbdoc(Origin of the grid.)pbdoc")
        .def_property_readonly("gridSpacing", &GridSystemData2::gridSpacing,
                               R"pbdoc(Spacing between grid points.)pbdoc")
        .def_property_readonly("boundingBox", &GridSystemData2::boundingBox,
                               R"pbdoc(Bounding box of the entire grid.)pbdoc");
}

void addGridSystemData3(py::module& m) {
    py::class_<GridSystemData3, GridSystemData3Ptr, Serializable>(
        m, "GridSystemData3",
        R"pbdoc(
        3-D grid system data.

        This class is the key data structure for storing grid system data. To
        represent a grid system for fluid simulation, velocity field is defined as a
        face-centered (MAC) grid by default. It can also have additional scalar or
        vector attributes by adding extra data layer.
        )pbdoc")
        .def(py::init<>())
        .def_property_readonly("resolution", &GridSystemData3::resolution,
                               R"pbdoc(Resolution of the grid.)pbdoc")
        .def_property_readonly("origin", &GridSystemData3::origin,
                               R"pbdoc(Origin of the grid.)pbdoc")
        .def_property_readonly("gridSpacing", &GridSystemData3::gridSpacing,
                               R"pbdoc(Spacing between grid points.)pbdoc")
        .def_property_readonly("boundingBox", &GridSystemData3::boundingBox,
                               R"pbdoc(Bounding box of the entire grid.)pbdoc");
}
