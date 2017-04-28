// Copyright (c) 2017 Doyub Kim
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
    py::class_<GridSystemData2, GridSystemData2Ptr>(m, "GridSystemData2",
        R"pbdoc(
        2-D grid system data.

        This class is the key data structure for storing grid system data. To
        represent a grid system for fluid simulation, velocity field is defined as a
        face-centered (MAC) grid by default. It can also have additional scalar or
        vector attributes by adding extra data layer.
        )pbdoc");
}

void addGridSystemData3(py::module& m) {
    py::class_<GridSystemData3, GridSystemData3Ptr>(m, "GridSystemData3",
        R"pbdoc(
        3-D grid system data.

        This class is the key data structure for storing grid system data. To
        represent a grid system for fluid simulation, velocity field is defined as a
        face-centered (MAC) grid by default. It can also have additional scalar or
        vector attributes by adding extra data layer.
        )pbdoc");
}
