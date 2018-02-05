// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_blocked_boundary_condition_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/grid_blocked_boundary_condition_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridBlockedBoundaryConditionSolver2(py::module& m) {
    py::class_<GridBlockedBoundaryConditionSolver2,
            GridBlockedBoundaryConditionSolver2Ptr,
            GridBoundaryConditionSolver2>(
            m, "GridBlockedBoundaryConditionSolver2",
            R"pbdoc(
        Blocked 2-D boundary condition solver for grids.

        This class constrains the velocity field by projecting the flow to the
        blocked representation of the collider. A collider is rasterized into voxels
        and each face of the collider voxels projects the velocity field onto its
        face. This implementation should pair up with GridSinglePhasePressureSolver2
        since the pressure solver assumes blocked boundary representation as well.
        )pbdoc")
            .def(py::init<>())
            .def("constrainVelocity",
                 [](GridBlockedBoundaryConditionSolver2& instance,
                    FaceCenteredGrid2Ptr velocity, unsigned int extrapolationDepth) {
                     instance.constrainVelocity(velocity.get(), extrapolationDepth);
                 },
                 R"pbdoc(
             Constrains the velocity field to conform the collider boundary.

             Parameters
             ----------
             - velocity : Input and output velocity grid.
             - extrapolationDepth : Number of inner-collider grid cells that
                                    velocity will get extrapolated.
             )pbdoc",
                 py::arg("velocity"), py::arg("extrapolationDepth") = 5);
}

void addGridBlockedBoundaryConditionSolver3(py::module& m) {
    py::class_<GridBlockedBoundaryConditionSolver3,
               GridBlockedBoundaryConditionSolver3Ptr,
               GridBoundaryConditionSolver3>(
        m, "GridBlockedBoundaryConditionSolver3",
        R"pbdoc(
        Blocked 3-D boundary condition solver for grids.

        This class constrains the velocity field by projecting the flow to the
        blocked representation of the collider. A collider is rasterized into voxels
        and each face of the collider voxels projects the velocity field onto its
        face. This implementation should pair up with GridSinglePhasePressureSolver3
        since the pressure solver assumes blocked boundary representation as well.
        )pbdoc")
        .def(py::init<>())
        .def("constrainVelocity",
             [](GridBlockedBoundaryConditionSolver3& instance,
                FaceCenteredGrid3Ptr velocity, unsigned int extrapolationDepth) {
                 instance.constrainVelocity(velocity.get(), extrapolationDepth);
             },
             R"pbdoc(
             Constrains the velocity field to conform the collider boundary.

             Parameters
             ----------
             - velocity : Input and output velocity grid.
             - extrapolationDepth : Number of inner-collider grid cells that
                                    velocity will get extrapolated.
             )pbdoc",
             py::arg("velocity"), py::arg("extrapolationDepth") = 5);
}
