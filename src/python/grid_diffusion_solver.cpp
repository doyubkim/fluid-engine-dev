// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_diffusion_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_diffusion_solver2.h>
#include <jet/grid_diffusion_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridDiffusionSolver2(py::module& m) {
    py::class_<GridDiffusionSolver2, GridDiffusionSolver2Ptr>(m, "GridDiffusionSolver2",
        R"pbdoc(
        Abstract base class for 2-D grid-based diffusion equation solver.

        This class provides functions to solve the diffusion equation for different
        types of fields. The target equation can be written as
        \f$\frac{\partial f}{\partial t} = \mu\nabla^2 f\f$ where \f$\mu\f$ is
        the diffusion coefficient. The field \f$f\f$ can be either scalar or vector
        field.
        )pbdoc");
}

void addGridDiffusionSolver3(py::module& m) {
    py::class_<GridDiffusionSolver3, GridDiffusionSolver3Ptr>(m, "GridDiffusionSolver3",
        R"pbdoc(
        Abstract base class for 3-D grid-based diffusion equation solver.

        This class provides functions to solve the diffusion equation for different
        types of fields. The target equation can be written as
        \f$\frac{\partial f}{\partial t} = \mu\nabla^2 f\f$ where \f$\mu\f$ is
        the diffusion coefficient. The field \f$f\f$ can be either scalar or vector
        field.
        )pbdoc");
}
