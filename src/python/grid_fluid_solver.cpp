// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_fluid_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_fluid_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridFluidSolver3(py::module& m) {
    py::class_<GridFluidSolver3, GridFluidSolver3Ptr, PhysicsAnimation>(
        m, "GridFluidSolver3")
        .def_property("gravity", &GridFluidSolver3::gravity,
                      [](GridFluidSolver3& instance, py::object obj) {
                          instance.setGravity(objectToVector3D(obj));
                      })
        .def_property("viscosityCoefficient",
                      &GridFluidSolver3::viscosityCoefficient,
                      &GridFluidSolver3::setViscosityCoefficient)
        .def("cfl", &GridFluidSolver3::cfl)
        .def_property("maxCfl", &GridFluidSolver3::maxCfl,
                      &GridFluidSolver3::setMaxCfl)
        .def_property("closedDomainBoundaryFlag",
                      &GridFluidSolver3::closedDomainBoundaryFlag,
                      &GridFluidSolver3::setClosedDomainBoundaryFlag)
        .def("resizeGrid",
             [](GridFluidSolver3& instance, py::args args, py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};

                 parseGridResizeParams(args, kwargs, resolution, gridSpacing,
                                       gridOrigin);

                 instance.resizeGrid(resolution, gridSpacing, gridOrigin);
             })
        .def_property_readonly("resolution", &GridFluidSolver3::resolution)
        .def_property_readonly("gridSpacing", &GridFluidSolver3::gridSpacing)
        .def_property_readonly("gridOrigin", &GridFluidSolver3::gridOrigin)
        .def_property(
            "collider",
            [](const GridFluidSolver3& instance) {
                return instance.collider();
            },
            [](GridFluidSolver3& instance, const Collider3Ptr& collider) {
                instance.setCollider(collider);
            });
}
