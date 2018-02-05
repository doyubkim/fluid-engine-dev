// Copyright (c) 2018 Doyub Kim
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
             R"pbdoc(
             Constructs PicSolver2

             This method constructs PicSolver2 with resolution, gridSpacing,
             and gridOrigin.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def_property_readonly("particleSystemData",
                               &PicSolver2::particleSystemData,
                               R"pbdoc(Returns particleSystemData.)pbdoc")
        .def_property("particleEmitter", &PicSolver2::particleEmitter,
                      &PicSolver2::setParticleEmitter,
                      R"pbdoc(Particle emitter property.)pbdoc");
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
             R"pbdoc(
             Constructs PicSolver3

             This method constructs PicSolver3 with resolution, gridSpacing,
             and gridOrigin.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def_property_readonly("particleSystemData",
                               &PicSolver3::particleSystemData,
                               R"pbdoc(Returns particleSystemData.)pbdoc")
        .def_property("particleEmitter", &PicSolver3::particleEmitter,
                      &PicSolver3::setParticleEmitter,
                      R"pbdoc(Particle emitter property.)pbdoc");
}
