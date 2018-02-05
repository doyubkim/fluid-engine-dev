// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_fractional_single_phase_pressure_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridFractionalSinglePhasePressureSolver2(py::module& m) {
    py::class_<GridFractionalSinglePhasePressureSolver2,
               GridFractionalSinglePhasePressureSolver2Ptr,
               GridPressureSolver2>(m,
                                    "GridFractionalSinglePhasePressureSolver2",
                                    R"pbdoc(
        2-D fractional single-phase pressure solver.

        This class implements 2-D fractional (or variational) single-phase pressure
        solver. It is called fractional because the solver encodes the boundaries
        to the grid cells like anti-aliased pixels, meaning that a grid cell will
        record the partially overlapping boundary as a fractional number.
        Alternative approach is to represent boundaries like Lego blocks which is
        the case for GridSinglePhasePressureSolver2.
        In addition, this class solves single-phase flow, solving the pressure for
        selected fluid region only and treat other area as an atmosphere region.
        Thus, the pressure outside the fluid will be set to a constant value and
        velocity field won't be altered. This solver also computes the fluid
        boundary in fractional manner, meaning that the solver tries to capture the
        subgrid structures. This class uses ghost fluid method for such calculation.
        \see Batty, Christopher, Florence Bertails, and Robert Bridson.
            "A fast variational framework for accurate solid-fluid coupling."
            ACM Transactions on Graphics (TOG). Vol. 26. No. 2. ACM, 2007.
        \see Enright, Doug, et al. "Using the particle level set method and
            a second order accurate pressure boundary condition for free surface
            flows." ASME/JSME 2002 4th Joint Fluids Summer Engineering Conference.
            American Society of Mechanical Engineers, 2002.
        )pbdoc")
        .def(py::init<>())
        .def_property(
            "linearSystemSolver",
            &GridFractionalSinglePhasePressureSolver2::linearSystemSolver,
            &GridFractionalSinglePhasePressureSolver2::setLinearSystemSolver,
            R"pbdoc(
            "The linear system solver."
            )pbdoc");
}

void addGridFractionalSinglePhasePressureSolver3(py::module& m) {
    py::class_<GridFractionalSinglePhasePressureSolver3,
               GridFractionalSinglePhasePressureSolver3Ptr,
               GridPressureSolver3>(m,
                                    "GridFractionalSinglePhasePressureSolver3",
                                    R"pbdoc(
        3-D fractional single-phase pressure solver.

        This class implements 3-D fractional (or variational) single-phase pressure
        solver. It is called fractional because the solver encodes the boundaries
        to the grid cells like anti-aliased pixels, meaning that a grid cell will
        record the partially overlapping boundary as a fractional number.
        Alternative approach is to represent boundaries like Lego blocks which is
        the case for GridSinglePhasePressureSolver2.
        In addition, this class solves single-phase flow, solving the pressure for
        selected fluid region only and treat other area as an atmosphere region.
        Thus, the pressure outside the fluid will be set to a constant value and
        velocity field won't be altered. This solver also computes the fluid
        boundary in fractional manner, meaning that the solver tries to capture the
        subgrid structures. This class uses ghost fluid method for such calculation.
        \see Batty, Christopher, Florence Bertails, and Robert Bridson.
            "A fast variational framework for accurate solid-fluid coupling."
            ACM Transactions on Graphics (TOG). Vol. 26. No. 3. ACM, 2007.
        \see Enright, Doug, et al. "Using the particle level set method and
            a second order accurate pressure boundary condition for free surface
            flows." ASME/JSME 2003 4th Joint Fluids Summer Engineering Conference.
            American Society of Mechanical Engineers, 2003.
        )pbdoc")
        .def(py::init<>())
        .def_property(
            "linearSystemSolver",
            &GridFractionalSinglePhasePressureSolver3::linearSystemSolver,
            &GridFractionalSinglePhasePressureSolver3::setLinearSystemSolver,
            R"pbdoc(
            "The linear system solver."
            )pbdoc");
}
