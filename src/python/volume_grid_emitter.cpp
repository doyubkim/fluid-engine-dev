// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "volume_grid_emitter.h"
#include "pybind11_utils.h"

#include <jet/surface_to_implicit2.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_grid_emitter2.h>
#include <jet/volume_grid_emitter3.h>

namespace py = pybind11;
using namespace jet;

void addVolumeGridEmitter2(py::module& m) {
    py::class_<VolumeGridEmitter2, VolumeGridEmitter2Ptr, GridEmitter2>(
        m, "VolumeGridEmitter2",
        R"pbdoc(2-D grid-based volumetric emitter.)pbdoc")
        .def("__init__",
             [](VolumeGridEmitter2& instance, const Surface2Ptr& sourceRegion,
                bool isOneShot) {
                 ImplicitSurface3Ptr sourceRegion_;
                 auto implicit =
                     std::dynamic_pointer_cast<ImplicitSurface2>(sourceRegion);
                 if (implicit != nullptr) {
                     new (&instance) VolumeGridEmitter2(implicit, isOneShot);
                 } else {
                     new (&instance) VolumeGridEmitter2(
                         std::make_shared<SurfaceToImplicit2>(sourceRegion),
                         isOneShot);
                 }
             },
             R"pbdoc(
             Constructs an emitter with a source and is-one-shot flag.
             )pbdoc",
             py::arg("sourceRegion"), py::arg("isOneShot") = true)
        .def("addSignedDistanceTarget",
             &VolumeGridEmitter2::addSignedDistanceTarget,
             R"pbdoc(
             Adds signed-distance target to the scalar grid.
             )pbdoc",
             py::arg("scalarGridTarget"))
        .def("addStepFunctionTarget",
             &VolumeGridEmitter2::addStepFunctionTarget,
             R"pbdoc(
             Adds step function target to the scalar grid.

             Parameters
             ----------
             - scalarGridTarget : The scalar grid target.
             - minValue : The minimum value of the step function.
             - maxValue : The maximum value of the step function.
             )pbdoc",
             py::arg("scalarGridTarget"), py::arg("minValue"),
             py::arg("maxValue"))
        .def("addTarget",
             [](VolumeGridEmitter2& instance, py::object obj,
                py::function mapper) {
                 if (py::isinstance<ScalarGrid2Ptr>(obj)) {
                     instance.addTarget(
                         obj.cast<ScalarGrid2Ptr>(),
                         [mapper](double ds, const Vector2D& l,
                                  double old) -> double {
                             return mapper(ds, l, old).cast<double>();
                         });
                 } else if (py::isinstance<VectorGrid2Ptr>(obj)) {
                     instance.addTarget(
                         obj.cast<VectorGrid2Ptr>(),
                         [mapper](double ds, const Vector2D& l,
                                  const Vector2D& old) -> Vector2D {
                             return mapper(ds, l, old).cast<Vector2D>();
                         });
                 } else {
                     throw std::invalid_argument("Unknown grid type.");
                 }
             },
             R"pbdoc(
             Adds a scalar/vector grid target.

             This function adds a custom target to the emitter. The first parameter
             defines which grid should it write to. The second parameter,
             `customMapper`, defines how to map signed-distance field from the
             volume geometry and location of the point to the final value that
             is going to be written to the target grid. The third parameter defines
             how to blend the old value from the target grid and the new value from
             the mapper function.

             Parameters
             ----------
             - gridTarget : The scalar/vector grid target.
             - customMapper : The custom mapper.
             )pbdoc",
             py::arg("scalarGridTarget"), py::arg("customMapper"))
        .def_property_readonly("sourceRegion",
                               &VolumeGridEmitter2::sourceRegion,
                               R"pbdoc(
             Implicit surface which defines the source region.
             )pbdoc")
        .def_property_readonly("isOneShot", &VolumeGridEmitter2::isOneShot,
                               R"pbdoc(
             True if this emits only once.
             )pbdoc");
}

void addVolumeGridEmitter3(py::module& m) {
    py::class_<VolumeGridEmitter3, VolumeGridEmitter3Ptr, GridEmitter3>(
        m, "VolumeGridEmitter3",
        R"pbdoc(3-D grid-based volumetric emitter.)pbdoc")
        .def("__init__",
             [](VolumeGridEmitter3& instance, const Surface3Ptr& sourceRegion,
                bool isOneShot) {
                 ImplicitSurface3Ptr sourceRegion_;
                 auto implicit =
                     std::dynamic_pointer_cast<ImplicitSurface3>(sourceRegion);
                 if (implicit != nullptr) {
                     new (&instance) VolumeGridEmitter3(implicit, isOneShot);
                 } else {
                     new (&instance) VolumeGridEmitter3(
                         std::make_shared<SurfaceToImplicit3>(sourceRegion),
                         isOneShot);
                 }
             },
             R"pbdoc(
             Constructs an emitter with a source and is-one-shot flag.
             )pbdoc",
             py::arg("sourceRegion"), py::arg("isOneShot") = true)
        .def("addSignedDistanceTarget",
             &VolumeGridEmitter3::addSignedDistanceTarget,
             R"pbdoc(
             Adds signed-distance target to the scalar grid.
             )pbdoc",
             py::arg("scalarGridTarget"))
        .def("addStepFunctionTarget",
             &VolumeGridEmitter3::addStepFunctionTarget,
             R"pbdoc(
             Adds step function target to the scalar grid.

             Parameters
             ----------
             - scalarGridTarget : The scalar grid target.
             - minValue : The minimum value of the step function.
             - maxValue : The maximum value of the step function.
             )pbdoc",
             py::arg("scalarGridTarget"), py::arg("minValue"),
             py::arg("maxValue"))
        .def("addTarget",
             [](VolumeGridEmitter3& instance, py::object obj,
                py::function mapper) {
                 if (py::isinstance<ScalarGrid3Ptr>(obj)) {
                     instance.addTarget(
                         obj.cast<ScalarGrid3Ptr>(),
                         [mapper](double ds, const Vector3D& l,
                                  double old) -> double {
                             return mapper(ds, l, old).cast<double>();
                         });
                 } else if (py::isinstance<VectorGrid3Ptr>(obj)) {
                     instance.addTarget(
                         obj.cast<VectorGrid3Ptr>(),
                         [mapper](double ds, const Vector3D& l,
                                  const Vector3D& old) -> Vector3D {
                             return mapper(ds, l, old).cast<Vector3D>();
                         });
                 } else {
                     throw std::invalid_argument("Unknown grid type.");
                 }
             },
             R"pbdoc(
             Adds a scalar/vector grid target.

             This function adds a custom target to the emitter. The first parameter
             defines which grid should it write to. The second parameter,
             `customMapper`, defines how to map signed-distance field from the
             volume geometry and location of the point to the final value that
             is going to be written to the target grid. The third parameter defines
             how to blend the old value from the target grid and the new value from
             the mapper function.

             Parameters
             ----------
             - gridTarget : The scalar/vector grid target.
             - customMapper : The custom mapper.
             )pbdoc",
             py::arg("scalarGridTarget"), py::arg("customMapper"))
        .def_property_readonly("sourceRegion",
                               &VolumeGridEmitter3::sourceRegion,
                               R"pbdoc(
             Implicit surface which defines the source region.
             )pbdoc")
        .def_property_readonly("isOneShot", &VolumeGridEmitter3::isOneShot,
                               R"pbdoc(
             True if this emits only once.
             )pbdoc");
}
