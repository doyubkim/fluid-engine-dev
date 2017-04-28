// Copyright (c) 2017 Doyub Kim
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
    py::class_<Grid2, Grid2Ptr>(m, "Grid2",
        R"pbdoc(
        Abstract base class for 2-D cartesian grid structure.

        This class represents 2-D cartesian grid structure. This class is an
        abstract base class and does not store any data. The class only stores the
        shape of the grid. The grid structure is axis-aligned and can have different
        grid spacing per axis.
        )pbdoc");
}

void addGrid3(py::module& m) {
    py::class_<Grid3, Grid3Ptr>(m, "Grid3",
        R"pbdoc(
        Abstract base class for 3-D cartesian grid structure.

        This class represents 3-D cartesian grid structure. This class is an
        abstract base class and does not store any data. The class only stores the
        shape of the grid. The grid structure is axis-aligned and can have different
        grid spacing per axis.
        )pbdoc");
}
