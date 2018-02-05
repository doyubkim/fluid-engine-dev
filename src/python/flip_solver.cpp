// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "flip_solver.h"
#include "pybind11_utils.h"

#include <jet/flip_solver2.h>
#include <jet/flip_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFlipSolver2(py::module& m) {
    py::class_<FlipSolver2, FlipSolver2Ptr, PicSolver2>(m, "FlipSolver2",
                                                        R"pbdoc(
        2-D Fluid-Implicit Particle (FLIP) implementation.

        This class implements 2-D Fluid-Implicit Particle (FLIP) solver from the
        SIGGRAPH paper, Zhu and Bridson 2005. By transfering delta-velocity field
        from grid to particles, the FLIP solver achieves less viscous fluid flow
        compared to the original PIC method.

        See: Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid."
            ACM Transactions on Graphics (TOG). Vol. 24. No. 2. ACM, 2005.
        )pbdoc")
        .def("__init__",
             [](FlipSolver2& instance, py::args args, py::kwargs kwargs) {
                 Size2 resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     FlipSolver2(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs FlipSolver2

             This method constructs FlipSolver2 with resolution, gridSpacing,
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
        .def_property("picBlendingFactor", &FlipSolver2::picBlendingFactor,
                      &FlipSolver2::setPicBlendingFactor,
                      R"pbdoc(
             The PIC blending factor.

             This property specifies the PIC blending factor which mixes FLIP and PIC
             results when transferring velocity from grids to particles in order to
             reduce the noise. The factor can be a value between 0 and 1, where 0
             means no blending and 1 means full PIC. Default is 0.
             )pbdoc");
}

void addFlipSolver3(py::module& m) {
    py::class_<FlipSolver3, FlipSolver3Ptr, PicSolver3>(m, "FlipSolver3",
                                                        R"pbdoc(
        3-D Fluid-Implicit Particle (FLIP) implementation.

        This class implements 3-D Fluid-Implicit Particle (FLIP) solver from the
        SIGGRAPH paper, Zhu and Bridson 2005. By transfering delta-velocity field
        from grid to particles, the FLIP solver achieves less viscous fluid flow
        compared to the original PIC method.

        See: Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid."
            ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
        )pbdoc")
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
             R"pbdoc(
             Constructs FlipSolver3

             This method constructs FlipSolver3 with resolution, gridSpacing,
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
        .def_property("picBlendingFactor", &FlipSolver3::picBlendingFactor,
                      &FlipSolver3::setPicBlendingFactor,
                      R"pbdoc(
             The PIC blending factor.

             This property specifies the PIC blending factor which mixes FLIP and PIC
             results when transferring velocity from grids to particles in order to
             reduce the noise. The factor can be a value between 0 and 1, where 0
             means no blending and 1 means full PIC. Default is 0.
             )pbdoc");
}
