// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "flip_solver3.h"
#include "pybind11_utils.h"

#include <jet/flip_solver3.h>

namespace py = pybind11;
using namespace jet;

void addFlipSolver3(py::module& m) {
    py::class_<FlipSolver3, FlipSolver3Ptr>(m, "FlipSolver3")
        // CTOR
        .def("__init__",
             [](FlipSolver3& instance, py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};
                 if (kwargs.contains("resolution")) {
                     resolution = tupleToSize3(py::tuple(kwargs["resolution"]));
                 }
                 if (kwargs.contains("gridSpacing")) {
                     gridSpacing =
                         tupleToVector3D(py::tuple(kwargs["gridSpacing"]));
                 }
                 if (kwargs.contains("gridOrigin")) {
                     gridOrigin =
                         tupleToVector3D(py::tuple(kwargs["gridOrigin"]));
                 }
                 new (&instance)
                     FlipSolver3(resolution, gridSpacing, gridOrigin);
             })
        // Animation
        .def("update", [](FlipSolver3& instance,
                          const Frame& frame) { instance.update(frame); })
        // PhysicsAnimation
        .def_property("isUsingFixedSubTimeSteps",
                      [](const FlipSolver3& instance) {
                          return instance.isUsingFixedSubTimeSteps();
                      },
                      [](FlipSolver3& instance, bool val) {
                          instance.setIsUsingFixedSubTimeSteps(val);
                      })
        .def_property("numberOfFixedSubTimeSteps",
                      [](const FlipSolver3& instance) {
                          return instance.numberOfFixedSubTimeSteps();
                      },
                      [](FlipSolver3& instance, unsigned int val) {
                          instance.setNumberOfFixedSubTimeSteps(val);
                      })
        .def("advanceSingleFrame",
             [](FlipSolver3& instance) { instance.advanceSingleFrame(); })
        .def_property(
            "currentFrame",
            [](const FlipSolver3& instance) { return instance.currentFrame(); },
            [](FlipSolver3& instance, const Frame& frame) {
                instance.setCurrentFrame(frame);
            })
        .def_property_readonly("currentTimeInSeconds",
                               [](const FlipSolver3& instance) {
                                   return instance.currentTimeInSeconds();
                               })
        // GridFluidSolver3
        .def_property("gravity",
                      [](const FlipSolver3& instance) {
                          return vector3ToTuple(instance.gravity());
                      },
                      [](FlipSolver3& instance, py::tuple val) {
                          instance.setGravity(tupleToVector3D(val));
                      })
        .def_property("viscosityCoefficient",
                      &FlipSolver3::viscosityCoefficient,
                      &FlipSolver3::setViscosityCoefficient)
        .def("cfl", &FlipSolver3::cfl)
        .def_property(
            "maxCfl",
            [](const FlipSolver3& instance) { return instance.maxCfl(); },
            [](FlipSolver3& instance, double val) { instance.setMaxCfl(val); })
        .def_property("closedDomainBoundaryFlag",
                      [](const FlipSolver3& instance) {
                          return instance.closedDomainBoundaryFlag();
                      },
                      [](FlipSolver3& instance, int val) {
                          instance.setClosedDomainBoundaryFlag(val);
                      })
        .def("resizeGrid",
             [](FlipSolver3& instance, py::kwargs kwargs) {
                 Size3 resolution{1, 1, 1};
                 Vector3D gridSpacing{1, 1, 1};
                 Vector3D gridOrigin{0, 0, 0};
                 if (kwargs.contains("resolution")) {
                     resolution = tupleToSize3(py::tuple(kwargs["resolution"]));
                 }
                 if (kwargs.contains("gridSpacing")) {
                     gridSpacing =
                         tupleToVector3D(py::tuple(kwargs["gridSpacing"]));
                 }
                 if (kwargs.contains("gridOrigin")) {
                     gridOrigin =
                         tupleToVector3D(py::tuple(kwargs["gridOrigin"]));
                 }
                 instance.resizeGrid(resolution, gridSpacing, gridOrigin);
             })
        .def_property_readonly(
            "gridResolution",
            [](const FlipSolver3& instance) {
                return size3ToTuple(instance.gridResolution());
            })
        .def_property_readonly(
            "gridSpacing",
            [](const FlipSolver3& instance) {
                return vector3ToTuple(instance.gridSpacing());
            })
        .def_property_readonly("gridOrigin",
                               [](const FlipSolver3& instance) {
                                   return vector3ToTuple(instance.gridOrigin());
                               })
        // FlipSolver3
        .def_property("picBlendingFactor", &FlipSolver3::picBlendingFactor,
                      &FlipSolver3::setPicBlendingFactor);
}
