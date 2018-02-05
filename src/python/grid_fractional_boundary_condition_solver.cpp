// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_fractional_boundary_condition_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/grid_fractional_boundary_condition_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridFractionalBoundaryConditionSolver2(py::module& m) {
    py::class_<GridFractionalBoundaryConditionSolver2,
               GridFractionalBoundaryConditionSolver2Ptr,
               GridBoundaryConditionSolver2>(
        m, "GridFractionalBoundaryConditionSolver2",
        R"pbdoc(
        Fractional 2-D boundary condition solver for grids.

        This class constrains the velocity field by projecting the flow to the
        signed-distance field representation of the collider. This implementation
        should pair up with GridFractionalSinglePhasePressureSolver2 to provide
        sub-grid resolutional velocity projection.
        )pbdoc")
        .def(py::init<>())
        .def(
            "constrainVelocity",
            [](GridFractionalBoundaryConditionSolver2& instance,
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
            py::arg("velocity"), py::arg("extrapolationDepth") = 5)
        .def_property_readonly(
            "colliderSdf", &GridFractionalBoundaryConditionSolver2::collider,
            R"pbdoc(
            Signed distance field of the collider.
            )pbdoc")
        .def_property_readonly(
            "colliderVelocityField",
            &GridFractionalBoundaryConditionSolver2::colliderVelocityField,
            R"pbdoc(
            Velocity field of the collider.
            )pbdoc");
}

void addGridFractionalBoundaryConditionSolver3(py::module& m) {
    py::class_<GridFractionalBoundaryConditionSolver3,
               GridFractionalBoundaryConditionSolver3Ptr,
               GridBoundaryConditionSolver3>(
        m, "GridFractionalBoundaryConditionSolver3",
        R"pbdoc(
        Fractional 3-D boundary condition solver for grids.

        This class constrains the velocity field by projecting the flow to the
        signed-distance field representation of the collider. This implementation
        should pair up with GridFractionalSinglePhasePressureSolver3 to provide
        sub-grid resolutional velocity projection.
        )pbdoc")
        .def(py::init<>())
        .def(
            "constrainVelocity",
            [](GridFractionalBoundaryConditionSolver3& instance,
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
            py::arg("velocity"), py::arg("extrapolationDepth") = 5)
        .def_property_readonly(
            "colliderSdf", &GridFractionalBoundaryConditionSolver3::collider,
            R"pbdoc(
            Signed distance field of the collider.
            )pbdoc")
        .def_property_readonly(
            "colliderVelocityField",
            &GridFractionalBoundaryConditionSolver3::colliderVelocityField,
            R"pbdoc(
            Velocity field of the collider.
            )pbdoc");
}
