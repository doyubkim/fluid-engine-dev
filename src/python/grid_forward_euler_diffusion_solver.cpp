// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_forward_euler_diffusion_solver.h"
#include "pybind11_utils.h"

#include <jet/grid_forward_euler_diffusion_solver2.h>
#include <jet/grid_forward_euler_diffusion_solver3.h>

namespace py = pybind11;
using namespace jet;

void addGridForwardEulerDiffusionSolver2(py::module& m) {
    py::class_<GridForwardEulerDiffusionSolver2,
               GridForwardEulerDiffusionSolver2Ptr, GridDiffusionSolver2>(
        m, "GridForwardEulerDiffusionSolver2",
        R"pbdoc(
        2-D grid-based forward Euler diffusion solver.

        This class implements 2-D grid-based forward Euler diffusion solver using
        second-order central differencing spatially. Since the method is relying on
        explicit time-integration (i.e. forward Euler), the diffusion coefficient is
        limited by the time interval and grid spacing such as:
        \f$\mu < \frac{h}{12\Delta t} \f$ where \f$\mu\f$, \f$h\f$, and
        \f$\Delta t\f$ are the diffusion coefficient, grid spacing, and time
        interval, respectively.
        )pbdoc")
        .def(py::init<>())
        .def("solver",
             [](GridForwardEulerDiffusionSolver2& instance, Grid2Ptr source,
                double diffusionCoefficient, double timeIntervalInSeconds,
                Grid2Ptr dest, py::kwargs kwargs) {
                 auto sourceSG = std::dynamic_pointer_cast<ScalarGrid2>(source);
                 auto sourceCG =
                     std::dynamic_pointer_cast<CollocatedVectorGrid2>(source);
                 auto sourceFG =
                     std::dynamic_pointer_cast<FaceCenteredGrid2>(source);

                 auto destSG = std::dynamic_pointer_cast<ScalarGrid2>(dest);
                 auto destCG =
                     std::dynamic_pointer_cast<CollocatedVectorGrid2>(dest);
                 auto destFG =
                     std::dynamic_pointer_cast<FaceCenteredGrid2>(dest);

                 ScalarField2Ptr boundarySdf = ConstantScalarField2::builder()
                                                   .withValue(kMaxD)
                                                   .makeShared();
                 ScalarField2Ptr fluidSdf = ConstantScalarField2::builder()
                                                .withValue(-kMaxD)
                                                .makeShared();

                 if (kwargs.contains("boundarySdf")) {
                     boundarySdf = kwargs.cast<ScalarField2Ptr>();
                 }
                 if (kwargs.contains("fluidSdf")) {
                     fluidSdf = kwargs.cast<ScalarField2Ptr>();
                 }

                 if (sourceSG != nullptr && destSG != nullptr) {
                     instance.solve(*sourceSG, diffusionCoefficient,
                                    timeIntervalInSeconds, destSG.get(),
                                    *boundarySdf, *fluidSdf);
                 } else if (sourceCG != nullptr && destCG != nullptr) {
                     instance.solve(*sourceCG, diffusionCoefficient,
                                    timeIntervalInSeconds, destCG.get(),
                                    *boundarySdf, *fluidSdf);
                 } else if (sourceFG != nullptr && destFG != nullptr) {
                     instance.solve(*sourceFG, diffusionCoefficient,
                                    timeIntervalInSeconds, destFG.get(),
                                    *boundarySdf, *fluidSdf);
                 } else {
                     throw std::invalid_argument(
                         "Grids source and dest must have same type.");
                 }
             },
             R"pbdoc(
             Solves diffusion equation for a scalar field.

             Parameters
             ----------
             - source : Input grid.
             - diffusionCoefficient : Amount of diffusion.
             - timeIntervalInSeconds : Small time-interval that diffusion occur.
             - dest : Output grid.
             - `**kwargs` :
                 - Key `boundarySdf` : Shape of the solid boundary that is empty by default.
                 - Key `fluidSdf` : Shape of the fluid boundary that is full by default.
             )pbdoc",
             py::arg("source"), py::arg("diffusionCoefficient"),
             py::arg("timeIntervalInSeconds"), py::arg("dest"));
}

void addGridForwardEulerDiffusionSolver3(py::module& m) {
    py::class_<GridForwardEulerDiffusionSolver3,
               GridForwardEulerDiffusionSolver3Ptr, GridDiffusionSolver3>(
        m, "GridForwardEulerDiffusionSolver3",
        R"pbdoc(
        3-D grid-based forward Euler diffusion solver.

        This class implements 3-D grid-based forward Euler diffusion solver using
        second-order central differencing spatially. Since the method is relying on
        explicit time-integration (i.e. forward Euler), the diffusion coefficient is
        limited by the time interval and grid spacing such as:
        \f$\mu < \frac{h}{12\Delta t} \f$ where \f$\mu\f$, \f$h\f$, and
        \f$\Delta t\f$ are the diffusion coefficient, grid spacing, and time
        interval, respectively.
        )pbdoc")
        .def(py::init<>())
        .def("solver",
             [](GridForwardEulerDiffusionSolver3& instance, Grid3Ptr source,
                double diffusionCoefficient, double timeIntervalInSeconds,
                Grid3Ptr dest, py::kwargs kwargs) {
                 auto sourceSG = std::dynamic_pointer_cast<ScalarGrid3>(source);
                 auto sourceCG =
                     std::dynamic_pointer_cast<CollocatedVectorGrid3>(source);
                 auto sourceFG =
                     std::dynamic_pointer_cast<FaceCenteredGrid3>(source);

                 auto destSG = std::dynamic_pointer_cast<ScalarGrid3>(dest);
                 auto destCG =
                     std::dynamic_pointer_cast<CollocatedVectorGrid3>(dest);
                 auto destFG =
                     std::dynamic_pointer_cast<FaceCenteredGrid3>(dest);

                 ScalarField3Ptr boundarySdf = ConstantScalarField3::builder()
                                                   .withValue(kMaxD)
                                                   .makeShared();
                 ScalarField3Ptr fluidSdf = ConstantScalarField3::builder()
                                                .withValue(-kMaxD)
                                                .makeShared();

                 if (kwargs.contains("boundarySdf")) {
                     boundarySdf = kwargs.cast<ScalarField3Ptr>();
                 }
                 if (kwargs.contains("fluidSdf")) {
                     fluidSdf = kwargs.cast<ScalarField3Ptr>();
                 }

                 if (sourceSG != nullptr && destSG != nullptr) {
                     instance.solve(*sourceSG, diffusionCoefficient,
                                    timeIntervalInSeconds, destSG.get(),
                                    *boundarySdf, *fluidSdf);
                 } else if (sourceCG != nullptr && destCG != nullptr) {
                     instance.solve(*sourceCG, diffusionCoefficient,
                                    timeIntervalInSeconds, destCG.get(),
                                    *boundarySdf, *fluidSdf);
                 } else if (sourceFG != nullptr && destFG != nullptr) {
                     instance.solve(*sourceFG, diffusionCoefficient,
                                    timeIntervalInSeconds, destFG.get(),
                                    *boundarySdf, *fluidSdf);
                 } else {
                     throw std::invalid_argument(
                         "Grids source and dest must have same type.");
                 }
             },
             R"pbdoc(
             Solves diffusion equation for a scalar field.

             Parameters
             ----------
             - source : Input grid.
             - diffusionCoefficient : Amount of diffusion.
             - timeIntervalInSeconds : Small time-interval that diffusion occur.
             - dest : Output grid.
             - `**kwargs` :
                 - Key `boundarySdf` : Shape of the solid boundary that is empty by default.
                 - Key `fluidSdf` : Shape of the fluid boundary that is full by default.
             )pbdoc",
             py::arg("source"), py::arg("diffusionCoefficient"),
             py::arg("timeIntervalInSeconds"), py::arg("dest"));
}
