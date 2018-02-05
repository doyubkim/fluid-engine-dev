// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_single_phase_pressure_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_single_phase_pressure_solver2.h>
#include <jet/grid_single_phase_pressure_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridSinglePhasePressureSolver2(py::module& m) {
    py::class_<GridSinglePhasePressureSolver2,
            GridSinglePhasePressureSolver2Ptr,
            GridPressureSolver2>(m,
                                 "GridSinglePhasePressureSolver2",
                                 R"pbdoc(
        2-D single-phase pressure solver.

        This class implements 2-D single-phase pressure solver. This solver encodes
        the boundaries like Lego blocks -- if a grid cell center is inside or
        outside the boundaries, it is either marked as occupied or not.
        In addition, this class solves single-phase flow, solving the pressure for
        selected fluid region only and treat other area as an atmosphere region.
        Thus, the pressure outside the fluid will be set to a constant value and
        velocity field won't be altered. This solver also computes the fluid
        boundary in block-like manner; If a grid cell is inside or outside the
        fluid, it is marked as either fluid or atmosphere. Thus, this solver in
        general, does not compute subgrid structure.
        )pbdoc")
            .def(py::init<>())
            .def_property(
                    "linearSystemSolver",
                    &GridSinglePhasePressureSolver2::linearSystemSolver,
                    &GridSinglePhasePressureSolver2::setLinearSystemSolver,
                    R"pbdoc(
            "The linear system solver."
            )pbdoc");
}

void addGridSinglePhasePressureSolver3(py::module& m) {
    py::class_<GridSinglePhasePressureSolver3,
               GridSinglePhasePressureSolver3Ptr,
               GridPressureSolver3>(m,
                                    "GridSinglePhasePressureSolver3",
                                    R"pbdoc(
        3-D single-phase pressure solver.

        This class implements 3-D single-phase pressure solver. This solver encodes
        the boundaries like Lego blocks -- if a grid cell center is inside or
        outside the boundaries, it is either marked as occupied or not.
        In addition, this class solves single-phase flow, solving the pressure for
        selected fluid region only and treat other area as an atmosphere region.
        Thus, the pressure outside the fluid will be set to a constant value and
        velocity field won't be altered. This solver also computes the fluid
        boundary in block-like manner; If a grid cell is inside or outside the
        fluid, it is marked as either fluid or atmosphere. Thus, this solver in
        general, does not compute subgrid structure.
        )pbdoc")
        .def(py::init<>())
        .def_property(
            "linearSystemSolver",
            &GridSinglePhasePressureSolver3::linearSystemSolver,
            &GridSinglePhasePressureSolver3::setLinearSystemSolver,
            R"pbdoc(
            "The linear system solver."
            )pbdoc");
}
