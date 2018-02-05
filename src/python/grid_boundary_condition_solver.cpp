// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_boundary_condition_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_boundary_condition_solver2.h>
#include <jet/grid_boundary_condition_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridBoundaryConditionSolver2(py::module& m) {
    py::class_<GridBoundaryConditionSolver2, GridBoundaryConditionSolver2Ptr>(
        m, "GridBoundaryConditionSolver2",
        R"pbdoc(
        Abstract base class for 2-D boundary condition solver for grids.

        This is a helper class to constrain the 2-D velocity field with given
        collider object. It also determines whether to open any domain boundaries.
        To control the friction level, tune the collider parameter.
        )pbdoc")
        .def_property_readonly("collider",
                               &GridBoundaryConditionSolver2::collider)
        .def("updateCollider",
             [](GridBoundaryConditionSolver2& instance,
                const Collider2Ptr& collider, py::object gridSize,
                py::object gridSpacing, py::object gridOrigin) {
                 instance.updateCollider(collider, objectToSize2(gridSize),
                                         objectToVector2D(gridSpacing),
                                         objectToVector2D(gridOrigin));
             },
             R"pbdoc(
             Applies new collider and build the internals.

             This function is called to apply new collider and build the internal
             cache. To provide a hint to the cache, info for the expected velocity
             grid that will be constrained is provided.

             Parameters
             ----------
             - collider : New collider to apply.
             - gridSize : Size of the velocity grid to be constrained.
             - gridSpacing : Grid spacing of the velocity grid to be constrained.
             - gridOrigin : Origin of the velocity grid to be constrained.
             )pbdoc",
             py::arg("collider"), py::arg("gridSize"), py::arg("gridSpacing"),
             py::arg("gridOrigin"))
        .def_property(
            "closedDomainBoundaryFlag",
            &GridBoundaryConditionSolver2::closedDomainBoundaryFlag,
            &GridBoundaryConditionSolver2::setClosedDomainBoundaryFlag,
            R"pbdoc(Closed domain boundary flag.)pbdoc");
}

void addGridBoundaryConditionSolver3(py::module& m) {
    py::class_<GridBoundaryConditionSolver3, GridBoundaryConditionSolver3Ptr>(
        m, "GridBoundaryConditionSolver3",
        R"pbdoc(
        Abstract base class for 3-D boundary condition solver for grids.

        This is a helper class to constrain the 3-D velocity field with given
        collider object. It also determines whether to open any domain boundaries.
        To control the friction level, tune the collider parameter.
        )pbdoc")
        .def_property_readonly("collider",
                               &GridBoundaryConditionSolver3::collider)
        .def("updateCollider",
             [](GridBoundaryConditionSolver3& instance,
                const Collider3Ptr& collider, py::object gridSize,
                py::object gridSpacing, py::object gridOrigin) {
                 instance.updateCollider(collider, objectToSize3(gridSize),
                                         objectToVector3D(gridSpacing),
                                         objectToVector3D(gridOrigin));
             },
             R"pbdoc(
             Applies new collider and build the internals.

             This function is called to apply new collider and build the internal
             cache. To provide a hint to the cache, info for the expected velocity
             grid that will be constrained is provided.

             Parameters
             ----------
             - collider : New collider to apply.
             - gridSize : Size of the velocity grid to be constrained.
             - gridSpacing : Grid spacing of the velocity grid to be constrained.
             - gridOrigin : Origin of the velocity grid to be constrained.
             )pbdoc",
             py::arg("collider"), py::arg("gridSize"), py::arg("gridSpacing"),
             py::arg("gridOrigin"))
        .def_property(
            "closedDomainBoundaryFlag",
            &GridBoundaryConditionSolver3::closedDomainBoundaryFlag,
            &GridBoundaryConditionSolver3::setClosedDomainBoundaryFlag,
            R"pbdoc(Closed domain boundary flag.)pbdoc");
}
