// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "iterative_level_set_solver.h"
#include "pybind11_utils.h"

#include <jet/iterative_level_set_solver2.h>
#include <jet/iterative_level_set_solver3.h>

namespace py = pybind11;
using namespace jet;

void addIterativeLevelSetSolver2(py::module& m) {
    py::class_<IterativeLevelSetSolver2, IterativeLevelSetSolver2Ptr,
               LevelSetSolver2>(m, "IterativeLevelSetSolver2",
                                R"pbdoc(
         Abstract base class for 2-D PDE-based iterative level set solver.

         This class provides infrastructure for 2-D PDE-based iterative level set
         solver. Internally, the class implements upwind-style wave propagation and
         the inheriting classes must provide a way to compute the derivatives for
         given grid points.

         - See Osher, Stanley, and Ronald Fedkiw. Level set methods and dynamic
               implicit surfaces. Vol. 153. Springer Science & Business Media, 2006.
         )pbdoc")
        .def("reinitialize",
             [](IterativeLevelSetSolver2& instance,
                const ScalarGrid2Ptr& inputSdf, double maxDistance,
                ScalarGrid2Ptr outputSdf) {
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
            [](IterativeLevelSetSolver2& instance, const Grid2Ptr& input,
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
            py::arg("output"))
        .def_property("maxCfl", &IterativeLevelSetSolver2::maxCfl,
                      &IterativeLevelSetSolver2::setMaxCfl,
                      R"pbdoc(The maximum CFL limit.)pbdoc");
}

void addIterativeLevelSetSolver3(py::module& m) {
    py::class_<IterativeLevelSetSolver3, IterativeLevelSetSolver3Ptr,
               LevelSetSolver3>(m, "IterativeLevelSetSolver3",
                                R"pbdoc(
         Abstract base class for 3-D PDE-based iterative level set solver.

         This class provides infrastructure for 3-D PDE-based iterative level set
         solver. Internally, the class implements upwind-style wave propagation and
         the inheriting classes must provide a way to compute the derivatives for
         given grid points.

         - See Osher, Stanley, and Ronald Fedkiw. Level set methods and dynamic
               implicit surfaces. Vol. 153. Springer Science & Business Media, 2006.
         )pbdoc")
        .def("reinitialize",
             [](IterativeLevelSetSolver3& instance,
                const ScalarGrid3Ptr& inputSdf, double maxDistance,
                ScalarGrid3Ptr outputSdf) {
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
            [](IterativeLevelSetSolver3& instance, const Grid3Ptr& input,
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
            py::arg("output"))
        .def_property("maxCfl", &IterativeLevelSetSolver3::maxCfl,
                      &IterativeLevelSetSolver3::setMaxCfl,
                      R"pbdoc(The maximum CFL limit.)pbdoc");
}
