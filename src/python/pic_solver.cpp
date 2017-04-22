// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pic_solver.h"
#include "pybind11_utils.h"

#include <jet/pic_solver3.h>

namespace py = pybind11;
using namespace jet;

void addPicSolver3(py::module& m) {
    py::class_<PicSolver3, PicSolver3Ptr, GridFluidSolver3>(m, "PicSolver3")
        .def("__init__",
             [](PicSolver3& instance, py::args args, py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     PicSolver3(resolution, gridSpacing, gridOrigin);
             },
             "Constructs PicSolver3\n\n"
             "This method constructs PicSolver3 with resolution, gridSpacing, "
             "and gridOrigin.")
        .def_property("particleEmitter", &PicSolver3::particleEmitter,
                      &PicSolver3::setParticleEmitter);
}
