// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "flip_solver.h"
#include "pybind11_utils.h"

#include <jet/flip_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFlipSolver3(py::module& m) {
    py::class_<FlipSolver3, FlipSolver3Ptr, PicSolver3>(m, "FlipSolver3")
        .def("__init__",
             [](FlipSolver3& instance, py::args args, py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     FlipSolver3(resolution, gridSpacing, gridOrigin);
             },
             "Constructs FlipSolver3\n\n"
             "This method constructs FlipSolver3 with resolution, gridSpacing, "
             "and gridOrigin.")
        .def_property("picBlendingFactor", &FlipSolver3::picBlendingFactor,
                      &FlipSolver3::setPicBlendingFactor);
}
