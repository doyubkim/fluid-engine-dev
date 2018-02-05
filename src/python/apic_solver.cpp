// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "apic_solver.h"
#include "pybind11_utils.h"

#include <jet/apic_solver2.h>
#include <jet/apic_solver3.h>

namespace py = pybind11;
using namespace jet;

void addApicSolver2(py::module& m) {
    py::class_<ApicSolver2, ApicSolver2Ptr, PicSolver2>(m, "ApicSolver2",
                                                        R"pbdoc(
        2-D Affine Particle-in-Cell (APIC) implementation

        This class implements 2-D Affine Particle-in-Cell (APIC) solver from the
        SIGGRAPH paper, Jiang 2015.

        See: Jiang, Chenfanfu, et al. "The affine particle-in-cell method."
             ACM Transactions on Graphics (TOG) 34.4 (2015): 51.
        )pbdoc")
        .def("__init__",
             [](ApicSolver2& instance, py::args args, py::kwargs kwargs) {
                 Size2 resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     ApicSolver2(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs ApicSolver2

             This method constructs ApicSolver2 with resolution, gridSpacing,
             and gridOrigin.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc");
}

void addApicSolver3(py::module& m) {
    py::class_<ApicSolver3, ApicSolver3Ptr, PicSolver3>(m, "ApicSolver3",
                                                        R"pbdoc(
        3-D Affine Particle-in-Cell (APIC) implementation

        This class implements 3-D Affine Particle-in-Cell (APIC) solver from the
        SIGGRAPH paper, Jiang 2015.

        See: Jiang, Chenfanfu, et al. "The affine particle-in-cell method."
             ACM Transactions on Graphics (TOG) 34.4 (2015): 51.
        )pbdoc")
        .def("__init__",
             [](ApicSolver3& instance, py::args args, py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     ApicSolver3(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs ApicSolver3

             This method constructs ApicSolver3 with resolution, gridSpacing,
             and gridOrigin.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc");
}
