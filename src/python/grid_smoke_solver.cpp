// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_smoke_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_smoke_solver2.h>
#include <jet/grid_smoke_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridSmokeSolver2(py::module& m) {
    py::class_<GridSmokeSolver2, GridSmokeSolver2Ptr, GridFluidSolver2>(
        m, "GridSmokeSolver2", R"pbdoc(
        2-D grid-based smoke solver.

        This class extends GridFluidSolver2 to implement smoke simulation solver.
        It adds smoke density and temperature fields to define the smoke and uses
        buoyancy force to simulate hot rising smoke.

        See Fedkiw, Ronald, Jos Stam, and Henrik Wann Jensen.
            "Visual simulation of smoke." Proceedings of the 28th annual conference
            on Computer graphics and interactive techniques. ACM, 2001.
        )pbdoc")
        .def("__init__",
             [](GridSmokeSolver2& instance, py::args args, py::kwargs kwargs) {
                 Size2 resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     GridSmokeSolver2(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs GridSmokeSolver2

             This method constructs GridSmokeSolver2 with resolution, gridSpacing,
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
        .def_property("smokeDiffusionCoefficient",
                      &GridSmokeSolver2::smokeDiffusionCoefficient,
                      &GridSmokeSolver2::setSmokeDiffusionCoefficient,
                      R"pbdoc(The smoke diffusion coefficient.)pbdoc")
        .def_property("temperatureDiffusionCoefficient",
                      &GridSmokeSolver2::temperatureDiffusionCoefficient,
                      &GridSmokeSolver2::setTemperatureDiffusionCoefficient,
                      R"pbdoc(The temperature diffusion coefficient.)pbdoc")
        .def_property("buoyancySmokeDensityFactor",
                      &GridSmokeSolver2::buoyancySmokeDensityFactor,
                      &GridSmokeSolver2::setBuoyancySmokeDensityFactor,
                      R"pbdoc(
                      The buoyancy factor which will be multiplied to the smoke density.

                      This class computes buoyancy by looking up the value of smoke density
                      and temperature, compare them to the average values, and apply
                      multiplier factor to the diff between the value and the average. That
                      multiplier is defined for each smoke density and temperature separately
                      For example, negative smoke density buoyancy factor means a heavier
                      smoke should sink.
                      )pbdoc")
        .def_property("buoyancyTemperatureFactor",
                      &GridSmokeSolver2::buoyancyTemperatureFactor,
                      &GridSmokeSolver2::setBuoyancyTemperatureFactor,
                      R"pbdoc(
                      The buoyancy factor which will be multiplied to the temperature.

                      This class computes buoyancy by looking up the value of smoke density
                      and temperature, compare them to the average values, and apply
                      multiplier factor to the diff between the value and the average. That
                      multiplier is defined for each smoke density and temperature separately
                      For example, negative smoke density buoyancy factor means a heavier
                      smoke should sink.
                      )pbdoc")
        .def_property("smokeDecayFactor", &GridSmokeSolver2::smokeDecayFactor,
                      &GridSmokeSolver2::setSmokeDecayFactor,
                      R"pbdoc(
                      The smoke decay factor.

                      In addition to the diffusion, the smoke also can fade-out over time by
                      setting the decay factor between 0 and 1.
                      )pbdoc")
        .def_property("smokeTemperatureDecayFactor",
                      &GridSmokeSolver2::smokeTemperatureDecayFactor,
                      &GridSmokeSolver2::setTemperatureDecayFactor,
                      R"pbdoc(
                      The temperature decay factor.

                      In addition to the diffusion, the temperature also can fade-out over
                      time by setting the decay factor between 0 and 1.
                      )pbdoc")
        .def_property_readonly("smokeDensity", &GridSmokeSolver2::smokeDensity,
                               R"pbdoc(Returns smoke density field.)pbdoc")
        .def_property_readonly("temperature", &GridSmokeSolver2::temperature,
                               R"pbdoc(Returns temperature field.)pbdoc");
}

void addGridSmokeSolver3(py::module& m) {
    py::class_<GridSmokeSolver3, GridSmokeSolver3Ptr, GridFluidSolver3>(
        m, "GridSmokeSolver3", R"pbdoc(
        3-D grid-based smoke solver.

        This class extends GridFluidSolver3 to implement smoke simulation solver.
        It adds smoke density and temperature fields to define the smoke and uses
        buoyancy force to simulate hot rising smoke.

        See Fedkiw, Ronald, Jos Stam, and Henrik Wann Jensen.
            "Visual simulation of smoke." Proceedings of the 28th annual conference
            on Computer graphics and interactive techniques. ACM, 2001.
        )pbdoc")
        .def("__init__",
             [](GridSmokeSolver3& instance, py::args args, py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 new (&instance)
                     GridSmokeSolver3(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Constructs GridSmokeSolver3

             This method constructs GridSmokeSolver3 with resolution, gridSpacing,
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
        .def_property("smokeDiffusionCoefficient",
                      &GridSmokeSolver3::smokeDiffusionCoefficient,
                      &GridSmokeSolver3::setSmokeDiffusionCoefficient,
                      R"pbdoc(The smoke diffusion coefficient.)pbdoc")
        .def_property("temperatureDiffusionCoefficient",
                      &GridSmokeSolver3::temperatureDiffusionCoefficient,
                      &GridSmokeSolver3::setTemperatureDiffusionCoefficient,
                      R"pbdoc(The temperature diffusion coefficient.)pbdoc")
        .def_property("buoyancySmokeDensityFactor",
                      &GridSmokeSolver3::buoyancySmokeDensityFactor,
                      &GridSmokeSolver3::setBuoyancySmokeDensityFactor,
                      R"pbdoc(
                      The buoyancy factor which will be multiplied to the smoke density.

                      This class computes buoyancy by looking up the value of smoke density
                      and temperature, compare them to the average values, and apply
                      multiplier factor to the diff between the value and the average. That
                      multiplier is defined for each smoke density and temperature separately
                      For example, negative smoke density buoyancy factor means a heavier
                      smoke should sink.
                      )pbdoc")
        .def_property("buoyancyTemperatureFactor",
                      &GridSmokeSolver3::buoyancyTemperatureFactor,
                      &GridSmokeSolver3::setBuoyancyTemperatureFactor,
                      R"pbdoc(
                      The buoyancy factor which will be multiplied to the temperature.

                      This class computes buoyancy by looking up the value of smoke density
                      and temperature, compare them to the average values, and apply
                      multiplier factor to the diff between the value and the average. That
                      multiplier is defined for each smoke density and temperature separately
                      For example, negative smoke density buoyancy factor means a heavier
                      smoke should sink.
                      )pbdoc")
        .def_property("smokeDecayFactor", &GridSmokeSolver3::smokeDecayFactor,
                      &GridSmokeSolver3::setSmokeDecayFactor,
                      R"pbdoc(
                      The smoke decay factor.

                      In addition to the diffusion, the smoke also can fade-out over time by
                      setting the decay factor between 0 and 1.
                      )pbdoc")
        .def_property("smokeTemperatureDecayFactor",
                      &GridSmokeSolver3::smokeTemperatureDecayFactor,
                      &GridSmokeSolver3::setTemperatureDecayFactor,
                      R"pbdoc(
                      The temperature decay factor.

                      In addition to the diffusion, the temperature also can fade-out over
                      time by setting the decay factor between 0 and 1.
                      )pbdoc")
        .def_property_readonly("smokeDensity", &GridSmokeSolver3::smokeDensity,
                               R"pbdoc(Returns smoke density field.)pbdoc")
        .def_property_readonly("temperature", &GridSmokeSolver3::temperature,
                               R"pbdoc(Returns temperature field.)pbdoc");
}
