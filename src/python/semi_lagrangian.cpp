// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "semi_lagrangian.h"
#include "pybind11_utils.h"

#include <jet/semi_lagrangian2.h>
#include <jet/semi_lagrangian3.h>

namespace py = pybind11;
using namespace jet;

void addSemiLagrangian2(py::module& m) {
    py::class_<SemiLagrangian2, SemiLagrangian2Ptr, AdvectionSolver2>(
        m, "SemiLagrangian2",
        R"pbdoc(
        Implementation of 2-D semi-Lagrangian advection solver.

        This class implements 2-D semi-Lagrangian advection solver. By default, the
        class implements 1st-order (linear) algorithm for the spatial interpolation.
        For the back-tracing, this class uses 2nd-order mid-point rule with adaptive
        time-stepping (CFL <= 1).
        )pbdoc")
        .def(py::init<>())
        .def("solve",
             [](SemiLagrangian2& instance, const Grid2Ptr& input,
                const VectorField2Ptr& flow, double dt, Grid2Ptr output,
                const ScalarField2Ptr& boundarySdf) {
                 auto inputSG = std::dynamic_pointer_cast<ScalarGrid2>(input);
                 auto inputCG =
                     std::dynamic_pointer_cast<CollocatedVectorGrid2>(input);
                 auto inputFG =
                     std::dynamic_pointer_cast<FaceCenteredGrid2>(input);

                 auto outputSG = std::dynamic_pointer_cast<ScalarGrid2>(output);
                 auto outputCG =
                     std::dynamic_pointer_cast<CollocatedVectorGrid2>(output);
                 auto outputFG =
                     std::dynamic_pointer_cast<FaceCenteredGrid2>(output);

                 if (inputSG != nullptr && outputSG != nullptr) {
                     instance.advect(*inputSG, *flow, dt, outputSG.get(),
                                     *boundarySdf);
                 } else if (inputCG != nullptr && outputCG != nullptr) {
                     instance.advect(*inputCG, *flow, dt, outputCG.get(),
                                     *boundarySdf);
                 } else if (inputFG != nullptr && outputFG != nullptr) {
                     instance.advect(*inputFG, *flow, dt, outputFG.get(),
                                     *boundarySdf);
                 } else {
                     throw std::invalid_argument(
                         "Grids input and output must have same type.");
                 }
             },
             R"pbdoc(
             Computes semi-Lagrangian for given scalar grid.

             This function computes semi-Lagrangian method to solve advection
             equation for given field `input` and underlying vector field
             `flow` that carries the input field. The solution after solving the
             equation for given time-step `dt` should be stored in field
             `output`. The boundary interface is given by a signed-distance field.
             The field is negative inside the boundary. By default, a constant field
             with max double value (kMaxD) is used, meaning no boundary.

             Parameters
             ----------
             - input : Input grid.
             - flow : Vector field that advects the input field.
             - dt : Time-step for the advection.
             - output : Output grid.
             - boundarySdf : Boundary interface defined by signed-distance field.
             )pbdoc",
             py::arg("input"), py::arg("flow"), py::arg("dt"),
             py::arg("output"),
             py::arg("boundarySdf") =
                 ConstantScalarField2::builder().withValue(kMaxD).makeShared());
}

void addSemiLagrangian3(py::module& m) {
    py::class_<SemiLagrangian3, SemiLagrangian3Ptr, AdvectionSolver3>(
        m, "SemiLagrangian3",
        R"pbdoc(
        Implementation of 3-D semi-Lagrangian advection solver.

        This class implements 3-D semi-Lagrangian advection solver. By default, the
        class implements 1st-order (linear) algorithm for the spatial interpolation.
        For the back-tracing, this class uses 3nd-order mid-point rule with adaptive
        time-stepping (CFL <= 1).
        )pbdoc")
        .def(py::init<>())
        .def("solve",
             [](SemiLagrangian3& instance, const Grid3Ptr& input,
                const VectorField3Ptr& flow, double dt, Grid3Ptr output,
                const ScalarField3Ptr& boundarySdf) {
                 auto inputSG = std::dynamic_pointer_cast<ScalarGrid3>(input);
                 auto inputCG =
                     std::dynamic_pointer_cast<CollocatedVectorGrid3>(input);
                 auto inputFG =
                     std::dynamic_pointer_cast<FaceCenteredGrid3>(input);

                 auto outputSG = std::dynamic_pointer_cast<ScalarGrid3>(output);
                 auto outputCG =
                     std::dynamic_pointer_cast<CollocatedVectorGrid3>(output);
                 auto outputFG =
                     std::dynamic_pointer_cast<FaceCenteredGrid3>(output);

                 if (inputSG != nullptr && outputSG != nullptr) {
                     instance.advect(*inputSG, *flow, dt, outputSG.get(),
                                     *boundarySdf);
                 } else if (inputCG != nullptr && outputCG != nullptr) {
                     instance.advect(*inputCG, *flow, dt, outputCG.get(),
                                     *boundarySdf);
                 } else if (inputFG != nullptr && outputFG != nullptr) {
                     instance.advect(*inputFG, *flow, dt, outputFG.get(),
                                     *boundarySdf);
                 } else {
                     throw std::invalid_argument(
                         "Grids input and output must have same type.");
                 }
             },
             R"pbdoc(
             Computes semi-Lagrangian for given scalar grid.

             This function computes semi-Lagrangian method to solve advection
             equation for given field `input` and underlying vector field
             `flow` that carries the input field. The solution after solving the
             equation for given time-step `dt` should be stored in field
             `output`. The boundary interface is given by a signed-distance field.
             The field is negative inside the boundary. By default, a constant field
             with max double value (kMaxD) is used, meaning no boundary.

             Parameters
             ----------
             - input : Input grid.
             - flow : Vector field that advects the input field.
             - dt : Time-step for the advection.
             - output : Output grid.
             - boundarySdf : Boundary interface defined by signed-distance field.
             )pbdoc",
             py::arg("input"), py::arg("flow"), py::arg("dt"),
             py::arg("output"),
             py::arg("boundarySdf") =
                 ConstantScalarField3::builder().withValue(kMaxD).makeShared());
}
