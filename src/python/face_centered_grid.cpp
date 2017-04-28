// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "face_centered_grid.h"
#include "pybind11_utils.h"

#include <jet/face_centered_grid2.h>
#include <jet/face_centered_grid3.h>

namespace py = pybind11;
using namespace jet;

void addFaceCenteredGrid2(py::module& m) {
    py::class_<FaceCenteredGrid2, FaceCenteredGrid2Ptr>(
        m, "FaceCenteredGrid2",
        R"pbdoc(
        2-D face-centered (a.k.a MAC or staggered) grid.

        This class implements face-centered grid which is also known as
        marker-and-cell (MAC) or staggered grid. This vector grid stores each vector
        component at face center. Thus, u and v components are not collocated.
        )pbdoc");
}

void addFaceCenteredGrid3(py::module& m) {
    py::class_<FaceCenteredGrid3, FaceCenteredGrid3Ptr>(
        m, "FaceCenteredGrid3",
        R"pbdoc(
        3-D face-centered (a.k.a MAC or staggered) grid.

        This class implements face-centered grid which is also known as
        marker-and-cell (MAC) or staggered grid. This vector grid stores each vector
        component at face center. Thus, u, v, and w components are not collocated.
        )pbdoc");
}
