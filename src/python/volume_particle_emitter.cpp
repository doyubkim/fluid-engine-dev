// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "volume_particle_emitter.h"
#include "pybind11_utils.h"

#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>

namespace py = pybind11;
using namespace jet;

void addVolumeParticleEmitter3(py::module& m) {
    py::class_<VolumeParticleEmitter3, VolumeParticleEmitter3Ptr,
               ParticleEmitter3>(m, "VolumeParticleEmitter3")
        // CTOR
        .def("__init__",
             [](VolumeParticleEmitter3& instance, py::args args,
                py::kwargs kwargs) {

                 ImplicitSurface3Ptr implicitSurface;
                 BoundingBox3D bounds;
                 double spacing = 0.1;
                 Vector3D initialVel;
                 size_t maxNumberOfParticles = kMaxSize;
                 double jitter = 0.0;
                 bool isOneShot = true;
                 bool allowOverlapping = false;
                 uint32_t seed = 0;

                 const auto parseImplicitSurface = [&](py::object arg) {
                     if (py::isinstance<ImplicitSurface3>(arg)) {
                         implicitSurface = arg.cast<ImplicitSurface3Ptr>();
                         bounds = implicitSurface->boundingBox();
                     } else if (py::isinstance<Surface3>(arg)) {
                         auto surface = arg.cast<Surface3Ptr>();
                         implicitSurface =
                             std::make_shared<SurfaceToImplicit3>(surface);
                         bounds = surface->boundingBox();
                     } else {
                         throw std::invalid_argument("Unknown type for implicitSurface.");
                     }
                 };

                 if (args.size() >= 3 && args.size() <= 9) {
                     parseImplicitSurface(args[0]);

                     bounds = args[1].cast<BoundingBox3D>();
                     spacing = args[2].cast<double>();

                     if (args.size() > 3) {
                         initialVel = objectToVector3D(py::object(args[3]));
                     }
                     if (args.size() > 4) {
                         maxNumberOfParticles = args[4].cast<size_t>();
                     }
                     if (args.size() > 5) {
                         jitter = args[5].cast<double>();
                     }
                     if (args.size() > 6) {
                         isOneShot = args[6].cast<bool>();
                     }
                     if (args.size() > 7) {
                         allowOverlapping = args[7].cast<bool>();
                     }
                     if (args.size() > 8) {
                         seed = args[8].cast<uint32_t>();
                     }
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 if (kwargs.contains("implicitSurface")) {
                     parseImplicitSurface(kwargs["implicitSurface"]);
                 }
                 if (kwargs.contains("bounds")) {
                     bounds = kwargs["bounds"].cast<BoundingBox3D>();
                 }
                 if (kwargs.contains("spacing")) {
                     spacing = kwargs["spacing"].cast<double>();
                 }
                 if (kwargs.contains("initialVel")) {
                     initialVel = objectToVector3D(kwargs["initialVel"]);
                 }
                 if (kwargs.contains("maxNumberOfParticles")) {
                     maxNumberOfParticles = kwargs["maxNumberOfParticles"].cast<size_t>();
                 }
                 if (kwargs.contains("jitter")) {
                     jitter = kwargs["jitter"].cast<double>();
                 }
                 if (kwargs.contains("isOneShot")) {
                     isOneShot = kwargs["isOneShot"].cast<bool>();
                 }
                 if (kwargs.contains("allowOverlapping")) {
                     allowOverlapping = kwargs["allowOverlapping"].cast<bool>();
                 }
                 if (kwargs.contains("seed")) {
                     seed = kwargs["seed"].cast<uint32_t>();
                 }

                 new (&instance) VolumeParticleEmitter3(
                     implicitSurface, bounds, spacing, initialVel,
                     maxNumberOfParticles, jitter, isOneShot, allowOverlapping,
                     seed);
             },
             "Constructs VolumeParticleEmitter3\n\n"
             "This method constructs VolumeParticleEmitter3 with implicit "
             "surface, bounding box, particle spacing, initial velocity "
             "(optional), max number of particles (optional), jitter "
             "(optional), whether it's one shot or not (optional), whether it "
             "should allow overlapping or not (optional), and random seed "
             "(optional).");
}
