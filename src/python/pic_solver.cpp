// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pic_solver.h"
#include "pybind11_utils.h"

#include <jet/pic_solver2.h>
#include <jet/pic_solver3.h>

namespace py = pybind11;
using namespace jet;

void addPicSolver2(py::module& m) {
    py::class_<PicSolver2, PicSolver2Ptr, GridFluidSolver2>(m, "PicSolver2")
        .def("__init__",
             [](PicSolver2& instance, py::args args, py::kwargs kwargs) {
                 Size2 resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     PicSolver2(resolution, gridSpacing, gridOrigin);
             },
             "Constructs PicSolver2\n\n"
             "This method constructs PicSolver2 with resolution, gridSpacing, "
             "and gridOrigin.")
        .def_property_readonly("particleSystemData",
                               &PicSolver2::particleSystemData)
        .def_property("particleEmitter", &PicSolver2::particleEmitter,
                      &PicSolver2::setParticleEmitter);
}

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
        .def_property_readonly("particleSystemData",
                               &PicSolver3::particleSystemData)
        .def_property("particleEmitter", &PicSolver3::particleEmitter,
                      &PicSolver3::setParticleEmitter);
}
