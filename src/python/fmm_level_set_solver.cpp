// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fmm_level_set_solver.h"
#include "pybind11_utils.h"

#include <jet/fmm_level_set_solver2.h>
#include <jet/fmm_level_set_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFmmLevelSetSolver2(py::module& m) {
    py::class_<FmmLevelSetSolver2, FmmLevelSetSolver2Ptr, LevelSetSolver2>(
        m, "FmmLevelSetSolver2",
        R"pbdoc(
         2-D fast marching method (FMM) implementation.

         This class implements 2-D FMM. First-order upwind-style differencing is used
         to solve the PDE.

         - See https://math.berkeley.edu/~sethian/2006/Explanations/fast_marching_explain.html
         - See Sethian, James A. "A fast marching level set method for monotonically
               advancing fronts." Proceedings of the National Academy of Sciences 93.4
               (1996): 1591-1595.
         )pbdoc")
        .def("reinitialize",
             [](FmmLevelSetSolver2& instance, const ScalarGrid2Ptr& inputSdf,
                double maxDistance, ScalarGrid2Ptr outputSdf) {
                 instance.reinitialize(*inputSdf, maxDistance, outputSdf.get());
             },
             R"pbdoc(
             Reinitializes given scalar field to signed-distance field.

             Parameters
             ----------
             - inputSdf : Input signed-distance field which can be distorted.
             - maxDistance : Max range of reinitialization.
             - outputSdf : Output signed-distance field.
             )pbdoc",
             py::arg("inputSdf"), py::arg("maxDistance"), py::arg("outputSdf"))
        .def(
            "extrapolate",
            [](FmmLevelSetSolver2& instance, const Grid2Ptr& input,
               const ScalarGrid2Ptr& sdf, double maxDistance, Grid2Ptr output) {
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
                    instance.extrapolate(*inputSG, *sdf, maxDistance,
                                         outputSG.get());
                } else if (inputCG != nullptr && outputCG != nullptr) {
                    instance.extrapolate(*inputCG, *sdf, maxDistance,
                                         outputCG.get());
                } else if (inputFG != nullptr && outputFG != nullptr) {
                    instance.extrapolate(*inputFG, *sdf, maxDistance,
                                         outputFG.get());
                } else {
                    throw std::invalid_argument(
                        "Grids input and output must have same type.");
                }
            },
            R"pbdoc(
             Extrapolates given field from negative to positive SDF region.

             Parameters
             ----------
             - input : Input field to be extrapolated.
             - sdf : Reference signed-distance field.
             - maxDistance : Max range of extrapolation.
             - output : Output field.
            )pbdoc",
            py::arg("input"), py::arg("sdf"), py::arg("maxDistance"),
            py::arg("output"));
}

void addFmmLevelSetSolver3(py::module& m) {
    py::class_<FmmLevelSetSolver3, FmmLevelSetSolver3Ptr, LevelSetSolver3>(
        m, "FmmLevelSetSolver3",
        R"pbdoc(
         3-D fast marching method (FMM) implementation.

         This class implements 3-D FMM. First-order upwind-style differencing is used
         to solve the PDE.

         - See https://math.berkeley.edu/~sethian/2006/Explanations/fast_marching_explain.html
         - See Sethian, James A. "A fast marching level set method for monotonically
               advancing fronts." Proceedings of the National Academy of Sciences 93.4
               (1996): 1591-1595.
         )pbdoc")
        .def("reinitialize",
             [](FmmLevelSetSolver3& instance, const ScalarGrid3Ptr& inputSdf,
                double maxDistance, ScalarGrid3Ptr outputSdf) {
                 instance.reinitialize(*inputSdf, maxDistance, outputSdf.get());
             },
             R"pbdoc(
             Reinitializes given scalar field to signed-distance field.

             Parameters
             ----------
             - inputSdf : Input signed-distance field which can be distorted.
             - maxDistance : Max range of reinitialization.
             - outputSdf : Output signed-distance field.
             )pbdoc",
             py::arg("inputSdf"), py::arg("maxDistance"), py::arg("outputSdf"))
        .def(
            "extrapolate",
            [](FmmLevelSetSolver3& instance, const Grid3Ptr& input,
               const ScalarGrid3Ptr& sdf, double maxDistance, Grid3Ptr output) {
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
                    instance.extrapolate(*inputSG, *sdf, maxDistance,
                                         outputSG.get());
                } else if (inputCG != nullptr && outputCG != nullptr) {
                    instance.extrapolate(*inputCG, *sdf, maxDistance,
                                         outputCG.get());
                } else if (inputFG != nullptr && outputFG != nullptr) {
                    instance.extrapolate(*inputFG, *sdf, maxDistance,
                                         outputFG.get());
                } else {
                    throw std::invalid_argument(
                        "Grids input and output must have same type.");
                }
            },
            R"pbdoc(
             Extrapolates given field from negative to positive SDF region.

             Parameters
             ----------
             - input : Input field to be extrapolated.
             - sdf : Reference signed-distance field.
             - maxDistance : Max range of extrapolation.
             - output : Output field.
            )pbdoc",
            py::arg("input"), py::arg("sdf"), py::arg("maxDistance"),
            py::arg("output"));
}
