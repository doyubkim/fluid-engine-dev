// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_fluid_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_fluid_solver2.h>
#include <jet/grid_fluid_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridFluidSolver2(py::module& m) {
    py::class_<GridFluidSolver2, GridFluidSolver2Ptr, PhysicsAnimation>(
        m, "GridFluidSolver2", R"pbdoc(
        Abstract base class for grid-based 2-D fluid solver.

        This is an abstract base class for grid-based 2-D fluid solver based on
        Jos Stam's famous 1999 paper - "Stable Fluids". This solver takes fractional
        step method as its foundation which is consisted of independent advection,
        diffusion, external forces, and pressure projection steps. Each step is
        configurable so that a custom step can be implemented. For example, if a
        user wants to change the advection solver to her/his own implementation,
        simply call setAdvectionSolver(newSolver).
        )pbdoc")
        .def_property("gravity", &GridFluidSolver2::gravity,
                      [](GridFluidSolver2& instance, py::object obj) {
                          instance.setGravity(objectToVector2D(obj));
                      },
                      R"pbdoc(The gravity vector of the system.)pbdoc")
        .def_property("viscosityCoefficient",
                      &GridFluidSolver2::viscosityCoefficient,
                      &GridFluidSolver2::setViscosityCoefficient,
                      R"pbdoc(The viscosity coefficient.)pbdoc")
        .def("cfl", &GridFluidSolver2::cfl,
             R"pbdoc(
             Returns the CFL number from the current velocity field for given
             time interval.

             Parameters
             ----------
             - timeIntervalInSeconds : The time interval in seconds.
             )pbdoc",
             py::arg("timeIntervalInSeconds"))
        .def_property("maxCfl", &GridFluidSolver2::maxCfl,
                      &GridFluidSolver2::setMaxCfl,
                      R"pbdoc(The max allowed CFL number.)pbdoc")
        .def_property(
            "useCompressedLinearSystem",
            &GridFluidSolver2::useCompressedLinearSystem,
            &GridFluidSolver2::setUseCompressedLinearSystem,
            R"pbdoc(True if the solver is using compressed linear system.)pbdoc")
        .def_property("advectionSolver", &GridFluidSolver2::advectionSolver,
                      &GridFluidSolver2::setAdvectionSolver,
                      R"pbdoc(The advection solver.)pbdoc")
        .def_property("diffusionSolver", &GridFluidSolver2::diffusionSolver,
                      &GridFluidSolver2::setDiffusionSolver,
                      R"pbdoc(The diffusion solver instance.)pbdoc")
        .def_property("pressureSolver", &GridFluidSolver2::pressureSolver,
                      &GridFluidSolver2::setPressureSolver,
                      R"pbdoc(The pressure solver instance.)pbdoc")
        .def_property("closedDomainBoundaryFlag",
                      &GridFluidSolver2::closedDomainBoundaryFlag,
                      &GridFluidSolver2::setClosedDomainBoundaryFlag,
                      R"pbdoc(
              The closed domain boundary flag.

              This flag is an integer which is bitwise-combination of
              DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_UP,
              and DIRECTION_BACK.
              )pbdoc")
        .def_property_readonly("gridSystemData",
                               &GridFluidSolver2::gridSystemData,
                               R"pbdoc(
             The grid system data.

             This function returns the grid system data. The grid system data stores
             the core fluid flow fields such as velocity. By default, the data
             instance has velocity field only.
             )pbdoc")
        .def("resizeGrid",
             [](GridFluidSolver2& instance, py::args args, py::kwargs kwargs) {
                 Size2 resolution{1, 1};
                 Vector2D gridSpacing{1, 1};
                 Vector2D gridOrigin{0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 instance.resizeGrid(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Resizes grid system data.

             This function resizes grid system data. You can also resize the grid by
             calling resize function directly from
             GridFluidSolver2::gridSystemData(), but this function provides a
             shortcut for the same operation.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def_property_readonly("resolution", &GridFluidSolver2::resolution,
                               R"pbdoc(
             The resolution of the grid system data.

             This function returns the resolution of the grid system data. This is
             equivalent to calling gridSystemData.resolution, but provides a
             shortcut.
             )pbdoc")
        .def_property_readonly("gridSpacing", &GridFluidSolver2::gridSpacing,
                               R"pbdoc(
             The grid spacing of the grid system data.

             This function returns the resolution of the grid system data. This is
             equivalent to calling gridSystemData.gridSpacing., but provides a
             shortcut.
             )pbdoc")
        .def_property_readonly("gridOrigin", &GridFluidSolver2::gridOrigin,
                               R"pbdoc(
             The origin of the grid system data.

             This function returns the resolution of the grid system data. This is
             equivalent to calling gridSystemData.origin., but provides a
             shortcut.
             )pbdoc")
        .def_property_readonly("velocity", &GridFluidSolver2::velocity,
                               R"pbdoc(
            The velocity field.

            This property returns the velocity field from the grid system data.
            It is just a shortcut to the most commonly accessed data chunk.
            )pbdoc")
        .def_property(
            "collider",
            [](const GridFluidSolver2& instance) {
                return instance.collider();
            },
            [](GridFluidSolver2& instance, const Collider2Ptr& collider) {
                instance.setCollider(collider);
            },
            R"pbdoc(The collider.)pbdoc")
        .def_property(
            "emitter",
            [](const GridFluidSolver2& instance) { return instance.emitter(); },
            [](GridFluidSolver2& instance, const GridEmitter2Ptr& emitter) {
                instance.setEmitter(emitter);
            },
            R"pbdoc(The emitter.)pbdoc");
}

void addGridFluidSolver3(py::module& m) {
    py::class_<GridFluidSolver3, GridFluidSolver3Ptr, PhysicsAnimation>(
        m, "GridFluidSolver3", R"pbdoc(
        Abstract base class for grid-based 3-D fluid solver.

        This is an abstract base class for grid-based 3-D fluid solver based on
        Jos Stam's famous 1999 paper - "Stable Fluids". This solver takes fractional
        step method as its foundation which is consisted of independent advection,
        diffusion, external forces, and pressure projection steps. Each step is
        configurable so that a custom step can be implemented. For example, if a
        user wants to change the advection solver to her/his own implementation,
        simply call setAdvectionSolver(newSolver).
        )pbdoc")
        .def_property("gravity", &GridFluidSolver3::gravity,
                      [](GridFluidSolver3& instance, py::object obj) {
                          instance.setGravity(objectToVector3D(obj));
                      },
                      R"pbdoc(The gravity vector of the system.)pbdoc")
        .def_property("viscosityCoefficient",
                      &GridFluidSolver3::viscosityCoefficient,
                      &GridFluidSolver3::setViscosityCoefficient,
                      R"pbdoc(The viscosity coefficient.)pbdoc")
        .def("cfl", &GridFluidSolver3::cfl,
             R"pbdoc(
             Returns the CFL number from the current velocity field for given
             time interval.

             Parameters
             ----------
             - timeIntervalInSeconds : The time interval in seconds.
             )pbdoc",
             py::arg("timeIntervalInSeconds"))
        .def_property("maxCfl", &GridFluidSolver3::maxCfl,
                      &GridFluidSolver3::setMaxCfl,
                      R"pbdoc(The max allowed CFL number.)pbdoc")
        .def_property(
            "useCompressedLinearSystem",
            &GridFluidSolver3::useCompressedLinearSystem,
            &GridFluidSolver3::setUseCompressedLinearSystem,
            R"pbdoc(True if the solver is using compressed linear system.)pbdoc")
        .def_property("advectionSolver", &GridFluidSolver3::advectionSolver,
                      &GridFluidSolver3::setAdvectionSolver,
                      R"pbdoc(The advection solver.)pbdoc")
        .def_property("diffusionSolver", &GridFluidSolver3::diffusionSolver,
                      &GridFluidSolver3::setDiffusionSolver,
                      R"pbdoc(The diffusion solver instance.)pbdoc")
        .def_property("pressureSolver", &GridFluidSolver3::pressureSolver,
                      &GridFluidSolver3::setPressureSolver,
                      R"pbdoc(The pressure solver instance.)pbdoc")
        .def_property("closedDomainBoundaryFlag",
                      &GridFluidSolver3::closedDomainBoundaryFlag,
                      &GridFluidSolver3::setClosedDomainBoundaryFlag,
                      R"pbdoc(
              The closed domain boundary flag.

              This flag is an integer which is bitwise-combination of
              DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_UP,
              DIRECTION_BACK, and DIRECTION_FRONT. Default is DIRECTION_ALL.
              )pbdoc")
        .def_property_readonly("gridSystemData",
                               &GridFluidSolver3::gridSystemData,
                               R"pbdoc(
             The grid system data.

             This function returns the grid system data. The grid system data stores
             the core fluid flow fields such as velocity. By default, the data
             instance has velocity field only.
             )pbdoc")
        .def("resizeGrid",
             [](GridFluidSolver3& instance, py::args args, py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 instance.resizeGrid(resolution, gridSpacing, gridOrigin);
             },
             R"pbdoc(
             Resizes grid system data.

             This function resizes grid system data. You can also resize the grid by
             calling resize function directly from
             GridFluidSolver2::gridSystemData(), but this function provides a
             shortcut for the same operation.

             Parameters
             ----------
             - `*args` : resolution, gridSpacing, and gridOrigin arguments.
             - `**kwargs`
                 - resolution : Grid resolution.
                 - gridSpacing : Grid spacing.
                 - gridOrigin : Origin point ot the grid.
                 - domainSizeX : Domain size in x-direction.
             )pbdoc")
        .def_property_readonly("resolution", &GridFluidSolver3::resolution,
                               R"pbdoc(
             The resolution of the grid system data.

             This function returns the resolution of the grid system data. This is
             equivalent to calling gridSystemData.resolution, but provides a
             shortcut.
             )pbdoc")
        .def_property_readonly("gridSpacing", &GridFluidSolver3::gridSpacing,
                               R"pbdoc(
             The grid spacing of the grid system data.

             This function returns the resolution of the grid system data. This is
             equivalent to calling gridSystemData.gridSpacing., but provides a
             shortcut.
             )pbdoc")
        .def_property_readonly("gridOrigin", &GridFluidSolver3::gridOrigin,
                               R"pbdoc(
             The origin of the grid system data.

             This function returns the resolution of the grid system data. This is
             equivalent to calling gridSystemData.origin., but provides a
             shortcut.
             )pbdoc")
        .def_property_readonly("velocity", &GridFluidSolver3::velocity,
                               R"pbdoc(
            The velocity field.

            This property returns the velocity field from the grid system data.
            It is just a shortcut to the most commonly accessed data chunk.
            )pbdoc")
        .def_property(
            "collider",
            [](const GridFluidSolver3& instance) {
                return instance.collider();
            },
            [](GridFluidSolver3& instance, const Collider3Ptr& collider) {
                instance.setCollider(collider);
            },
            R"pbdoc(The collider.)pbdoc")
        .def_property(
            "emitter",
            [](const GridFluidSolver3& instance) { return instance.emitter(); },
            [](GridFluidSolver3& instance, const GridEmitter3Ptr& emitter) {
                instance.setEmitter(emitter);
            },
            R"pbdoc(The emitter.)pbdoc");
}
