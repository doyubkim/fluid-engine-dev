// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "level_set_liquid_solver.h"
#include "pybind11_utils.h"

#include <jet/level_set_liquid_solver2.h>
#include <jet/level_set_liquid_solver3.h>

namespace py = pybind11;
using namespace jet;

void addLevelSetLiquidSolver2(py::module& m) {
    py::class_<LevelSetLiquidSolver2, LevelSetLiquidSolver2Ptr,
               GridFluidSolver2>(m, "LevelSetLiquidSolver2",
                                 R"pbdoc(
         Level set based 2-D liquid solver.

         This class implements level set-based 2-D liquid solver. It defines the
         surface of the liquid using signed-distance field and use stable fluids
         framework to compute the forces.

         - See Enright, Douglas, Stephen Marschner, and Ronald Fedkiw.
               "Animation and rendering of complex water surfaces." ACM Transactions on
               Graphics (TOG). Vol. 21. No. 3. ACM, 2002.
         )pbdoc")
        .def("__init__",
             [](LevelSetLiquidSolver2& instance, py::args args,
                py::kwargs kwargs) {
                 Size2 resolution{1, 1, 1};
                 Vector2D gridSpacing{1, 1, 1};
                 Vector2D gridOrigin{0, 0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     LevelSetLiquidSolver2(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs LevelSetLiquidSolver2

             This method constructs LevelSetLiquidSolver2 with resolution, gridSpacing,
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
        .def_property_readonly("signedDistanceField",
                               &LevelSetLiquidSolver2::signedDistanceField,
                               R"pbdoc(The signed-distance field.)pbdoc")
        .def_property("levelSetSolver", &LevelSetLiquidSolver2::levelSetSolver,
                      &LevelSetLiquidSolver2::setLevelSetSolver,
                      R"pbdoc(The level set solver.)pbdoc")
        .def("setMinReinitializeDistance",
             &LevelSetLiquidSolver2::setMinReinitializeDistance,
             R"pbdoc(Sets minimum reinitialization distance.)pbdoc")
        .def("setIsGlobalCompensationEnabled",
             &LevelSetLiquidSolver2::setIsGlobalCompensationEnabled,
             R"pbdoc(
             Enables (or disables) global compensation feature flag.

             When `isEnabled` is true, the global compensation feature is enabled.
             The global compensation measures the volume at the beginning and the end
             of the time-step and adds the volume change back to the level-set field
             by globally shifting the front.
             \see Song, Oh-Young, Hyuncheol Shin, and Hyeong-Seok Ko.
             "Stable but nondissipative water." ACM Transactions on Graphics (TOG)
             24, no. 1 (2005): 81-97.
             )pbdoc",
             py::arg("isEnabled"))
        .def("computeVolume", &LevelSetLiquidSolver2::computeVolume,
             R"pbdoc(
             Returns liquid volume measured by smeared Heaviside function.

             This function measures the liquid volume using smeared Heaviside
             function. Thus, the estimated volume is an approximated quantity.
             )pbdoc");
}

void addLevelSetLiquidSolver3(py::module& m) {
    py::class_<LevelSetLiquidSolver3, LevelSetLiquidSolver3Ptr,
               GridFluidSolver3>(m, "LevelSetLiquidSolver3",
                                 R"pbdoc(
         Level set based 3-D liquid solver.

         This class implements level set-based 3-D liquid solver. It defines the
         surface of the liquid using signed-distance field and use stable fluids
         framework to compute the forces.

         - See Enright, Douglas, Stephen Marschner, and Ronald Fedkiw.
               "Animation and rendering of complex water surfaces." ACM Transactions on
               Graphics (TOG). Vol. 21. No. 3. ACM, 2002.
         )pbdoc")
        .def("__init__",
             [](LevelSetLiquidSolver3& instance, py::args args,
                py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     LevelSetLiquidSolver3(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs LevelSetLiquidSolver3

             This method constructs LevelSetLiquidSolver3 with resolution, gridSpacing,
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
        .def_property_readonly("signedDistanceField",
                               &LevelSetLiquidSolver3::signedDistanceField,
                               R"pbdoc(The signed-distance field.)pbdoc")
        .def_property("levelSetSolver", &LevelSetLiquidSolver3::levelSetSolver,
                      &LevelSetLiquidSolver3::setLevelSetSolver,
                      R"pbdoc(The level set solver.)pbdoc")
        .def("setMinReinitializeDistance",
             &LevelSetLiquidSolver3::setMinReinitializeDistance,
             R"pbdoc(Sets minimum reinitialization distance.)pbdoc")
        .def("setIsGlobalCompensationEnabled",
             &LevelSetLiquidSolver3::setIsGlobalCompensationEnabled,
             R"pbdoc(
             Enables (or disables) global compensation feature flag.

             When `isEnabled` is true, the global compensation feature is enabled.
             The global compensation measures the volume at the beginning and the end
             of the time-step and adds the volume change back to the level-set field
             by globally shifting the front.
             \see Song, Oh-Young, Hyuncheol Shin, and Hyeong-Seok Ko.
             "Stable but nondissipative water." ACM Transactions on Graphics (TOG)
             24, no. 1 (2005): 81-97.
             )pbdoc",
             py::arg("isEnabled"))
        .def("computeVolume", &LevelSetLiquidSolver3::computeVolume,
             R"pbdoc(
             Returns liquid volume measured by smeared Heaviside function.

             This function measures the liquid volume using smeared Heaviside
             function. Thus, the estimated volume is an approximated quantity.
             )pbdoc");
}
