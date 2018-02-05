// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "vector_grid.h"
#include "pybind11_utils.h"

#include <jet/vector_grid2.h>
#include <jet/vector_grid3.h>

namespace py = pybind11;
using namespace jet;

void addVectorGrid2(py::module& m) {
    py::class_<VectorGrid2, VectorGrid2Ptr, VectorField2, Grid2>(
        m, "VectorGrid2",
        R"pbdoc(Abstract base class for 2-D vector grid structure.)pbdoc");
}

void addVectorGrid3(py::module& m) {
    py::class_<VectorGrid3, VectorGrid3Ptr, VectorField3, Grid3>(
        m, "VectorGrid3",
        R"pbdoc(Abstract base class for 3-D vector grid structure.)pbdoc");
}
