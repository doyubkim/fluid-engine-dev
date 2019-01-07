// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "volume_particle_emitter.h"
#include "pybind11_utils.h"

#include <jet/surface_to_implicit2.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter2.h>
#include <jet/volume_particle_emitter3.h>

namespace py = pybind11;
using namespace jet;

void addVolumeParticleEmitter2(py::module& m) {
    py::class_<VolumeParticleEmitter2, VolumeParticleEmitter2Ptr,
               ParticleEmitter2>(m, "VolumeParticleEmitter2")
        // CTOR
        .def("__init__",
             [](VolumeParticleEmitter2& instance, py::args args,
                py::kwargs kwargs) {

                 ImplicitSurface2Ptr implicitSurface;
                 BoundingBox2D maxRegion;
                 double spacing = 0.1;
                 Vector2D initialVel;
                 Vector2D linearVel;
                 double angularVel = 0.0;
                 size_t maxNumberOfParticles = kMaxSize;
                 double jitter = 0.0;
                 bool isOneShot = true;
                 bool allowOverlapping = false;
                 uint32_t seed = 0;

                 const auto parseImplicitSurface = [&](py::object arg) {
                     if (py::isinstance<ImplicitSurface2>(arg)) {
                         implicitSurface = arg.cast<ImplicitSurface2Ptr>();
                     } else if (py::isinstance<Surface2>(arg)) {
                         auto surface = arg.cast<Surface2Ptr>();
                         implicitSurface =
                             std::make_shared<SurfaceToImplicit2>(surface);
                     } else {
                         throw std::invalid_argument(
                             "Unknown type for implicitSurface.");
                     }

                     // Get initial val for the max region.
                     if (implicitSurface->isBounded()) {
                         maxRegion = implicitSurface->boundingBox();
                     }
                 };

                 if (args.size() >= 3 && args.size() <= 11) {
                     parseImplicitSurface(args[0]);

                     maxRegion = args[1].cast<BoundingBox2D>();
                     spacing = args[2].cast<double>();

                     if (args.size() > 3) {
                         initialVel = objectToVector2D(py::object(args[3]));
                     }
                     if (args.size() > 4) {
                         linearVel = objectToVector2D(py::object(args[4]));
                     }
                     if (args.size() > 5) {
                         angularVel = args[5].cast<double>();
                     }
                     if (args.size() > 6) {
                         maxNumberOfParticles = args[6].cast<size_t>();
                     }
                     if (args.size() > 7) {
                         jitter = args[7].cast<double>();
                     }
                     if (args.size() > 8) {
                         isOneShot = args[8].cast<bool>();
                     }
                     if (args.size() > 9) {
                         allowOverlapping = args[9].cast<bool>();
                     }
                     if (args.size() > 10) {
                         seed = args[10].cast<uint32_t>();
                     }
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 if (kwargs.contains("implicitSurface")) {
                     parseImplicitSurface(kwargs["implicitSurface"]);
                 }
                 // "bounds" will be deprecated in v2, but keeping here for v1.
                 if (kwargs.contains("bounds")) {
                     maxRegion = kwargs["bounds"].cast<BoundingBox2D>();
                 }
                 if (kwargs.contains("maxRegion")) {
                     maxRegion = kwargs["maxRegion"].cast<BoundingBox2D>();
                 }
                 if (kwargs.contains("spacing")) {
                     spacing = kwargs["spacing"].cast<double>();
                 }
                 if (kwargs.contains("initialVelocity")) {
                     initialVel = objectToVector2D(kwargs["initialVelocity"]);
                 }
                 if (kwargs.contains("linearVelocity")) {
                     linearVel = objectToVector2D(kwargs["linearVelocity"]);
                 }
                 if (kwargs.contains("angularVelocity")) {
                     angularVel = kwargs["angularVelocity"].cast<double>();
                 }
                 if (kwargs.contains("maxNumberOfParticles")) {
                     maxNumberOfParticles =
                         kwargs["maxNumberOfParticles"].cast<size_t>();
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

                 new (&instance) VolumeParticleEmitter2(
                     implicitSurface, maxRegion, spacing, initialVel, linearVel,
                     angularVel, maxNumberOfParticles, jitter, isOneShot,
                     allowOverlapping, seed);
             },
             R"pbdoc(
             Constructs VolumeParticleEmitter2

             This method constructs VolumeParticleEmitter2 with implicit
             surface, bounding box, particle spacing, initial velocity
             (optional), max number of particles (optional), jitter
             (optional), whether it's one shot or not (optional), whether it
             should allow overlapping or not (optional), and random seed
             (optional).

             Parameters
             ----------
             - implicitSurface : The implicit surface.
             - maxRegion: The max region.
             - spacing: The spacing between particles.
             - initialVel: The initial velocity.
             - maxNumberOfParticles: The max number of particles to be emitted.
             - jitter: The jitter amount between 0 and 1.
             - isOneShot: Set true if particles are emitted just once.
             - allowOverlapping: True if particles can be overlapped.
             - seed: The random seed.
             )pbdoc")
        .def_property("surface", &VolumeParticleEmitter2::surface,
                      &VolumeParticleEmitter2::setSurface, R"pbdoc(
             Source surface.
             )pbdoc")
        .def_property("maxRegion", &VolumeParticleEmitter2::maxRegion,
                      &VolumeParticleEmitter2::setMaxRegion, R"pbdoc(
             Max particle gen region.
             )pbdoc")
        .def_property("jitter", &VolumeParticleEmitter2::jitter,
                      &VolumeParticleEmitter2::setJitter, R"pbdoc(
             Jitter amount between 0 and 1.
             )pbdoc")
        .def_property("isOneShot", &VolumeParticleEmitter2::isOneShot,
                      &VolumeParticleEmitter2::setIsOneShot, R"pbdoc(
             True if particles should be emitted just once.
             )pbdoc")
        .def_property("allowOverlapping",
                      &VolumeParticleEmitter2::allowOverlapping,
                      &VolumeParticleEmitter2::setAllowOverlapping, R"pbdoc(
             True if particles can be overlapped.
             )pbdoc")
        .def_property(
            "maxNumberOfParticles", &VolumeParticleEmitter2::maxNumberOfParticles,
            &VolumeParticleEmitter2::setMaxNumberOfParticles, R"pbdoc(
             Max number of particles to be emitted.
             )pbdoc")
        .def_property("spacing", &VolumeParticleEmitter2::spacing,
                      &VolumeParticleEmitter2::setSpacing, R"pbdoc(
             The spacing between particles.
             )pbdoc")
        .def_property(
            "initialVelocity", &VolumeParticleEmitter2::initialVelocity,
            [](VolumeParticleEmitter2& instance, py::object newInitialVel) {
                instance.setInitialVelocity(objectToVector2D(newInitialVel));
            },
            R"pbdoc(
             The initial velocity of the particles.
             )pbdoc")
        .def_property(
            "linearVelocity", &VolumeParticleEmitter2::linearVelocity,
            [](VolumeParticleEmitter2& instance, py::object newLinearVel) {
                instance.setLinearVelocity(objectToVector2D(newLinearVel));
            },
            R"pbdoc(
             The linear velocity of the emitter.
             )pbdoc")
        .def_property(
            "angularVelocity", &VolumeParticleEmitter2::angularVelocity,
            [](VolumeParticleEmitter2& instance, double newAngularVel) {
                instance.setAngularVelocity(newAngularVel);
            },
            R"pbdoc(
             The angular velocity of the emitter.
             )pbdoc");
}

void addVolumeParticleEmitter3(py::module& m) {
    py::class_<VolumeParticleEmitter3, VolumeParticleEmitter3Ptr,
               ParticleEmitter3>(m, "VolumeParticleEmitter3")
        // CTOR
        .def("__init__",
             [](VolumeParticleEmitter3& instance, py::args args,
                py::kwargs kwargs) {

                 ImplicitSurface3Ptr implicitSurface;
                 BoundingBox3D maxRegion;
                 double spacing = 0.1;
                 Vector3D initialVel;
                 Vector3D linearVel;
                 Vector3D angularVel;
                 size_t maxNumberOfParticles = kMaxSize;
                 double jitter = 0.0;
                 bool isOneShot = true;
                 bool allowOverlapping = false;
                 uint32_t seed = 0;

                 const auto parseImplicitSurface = [&](py::object arg) {
                     if (py::isinstance<ImplicitSurface3>(arg)) {
                         implicitSurface = arg.cast<ImplicitSurface3Ptr>();
                     } else if (py::isinstance<Surface3>(arg)) {
                         auto surface = arg.cast<Surface3Ptr>();
                         implicitSurface =
                             std::make_shared<SurfaceToImplicit3>(surface);
                     } else {
                         throw std::invalid_argument(
                             "Unknown type for implicitSurface.");
                     }

                     // Get initial val for the max region.
                     if (implicitSurface->isBounded()) {
                         maxRegion = implicitSurface->boundingBox();
                     }
                 };

                 if (args.size() >= 3 && args.size() <= 11) {
                     parseImplicitSurface(args[0]);

                     maxRegion = args[1].cast<BoundingBox3D>();
                     spacing = args[2].cast<double>();

                     if (args.size() > 3) {
                         initialVel = objectToVector3D(py::object(args[3]));
                     }
                     if (args.size() > 4) {
                         linearVel = objectToVector3D(py::object(args[4]));
                     }
                     if (args.size() > 5) {
                         angularVel = objectToVector3D(py::object(args[5]));
                     }
                     if (args.size() > 6) {
                         maxNumberOfParticles = args[6].cast<size_t>();
                     }
                     if (args.size() > 7) {
                         jitter = args[7].cast<double>();
                     }
                     if (args.size() > 8) {
                         isOneShot = args[8].cast<bool>();
                     }
                     if (args.size() > 9) {
                         allowOverlapping = args[9].cast<bool>();
                     }
                     if (args.size() > 10) {
                         seed = args[10].cast<uint32_t>();
                     }
                 } else if (args.size() > 0) {
                     throw std::invalid_argument("Too few/many arguments.");
                 }

                 if (kwargs.contains("implicitSurface")) {
                     parseImplicitSurface(kwargs["implicitSurface"]);
                 }
                 // "bounds" will be deprecated in v2, but keeping here for v1.
                 if (kwargs.contains("bounds")) {
                     maxRegion = kwargs["bounds"].cast<BoundingBox3D>();
                 }
                 if (kwargs.contains("maxRegion")) {
                     maxRegion = kwargs["maxRegion"].cast<BoundingBox3D>();
                 }
                 if (kwargs.contains("spacing")) {
                     spacing = kwargs["spacing"].cast<double>();
                 }
                 if (kwargs.contains("initialVelocity")) {
                     initialVel = objectToVector3D(kwargs["initialVelocity"]);
                 }
                 if (kwargs.contains("linearVelocity")) {
                     linearVel = objectToVector3D(kwargs["linearVelocity"]);
                 }
                 if (kwargs.contains("angularVelocity")) {
                     angularVel = objectToVector3D(kwargs["angularVelocity"]);
                 }
                 if (kwargs.contains("maxNumberOfParticles")) {
                     maxNumberOfParticles =
                         kwargs["maxNumberOfParticles"].cast<size_t>();
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
                     implicitSurface, maxRegion, spacing, initialVel, linearVel,
                     angularVel, maxNumberOfParticles, jitter, isOneShot,
                     allowOverlapping, seed);
             },
             R"pbdoc(
             Constructs VolumeParticleEmitter3

             This method constructs VolumeParticleEmitter3 with implicit
             surface, bounding box, particle spacing, initial velocity
             (optional), max number of particles (optional), jitter
             (optional), whether it's one shot or not (optional), whether it
             should allow overlapping or not (optional), and random seed
             (optional).

             Parameters
             ----------
             - implicitSurface : The implicit surface.
             - maxRegion: The max region.
             - spacing: The spacing between particles.
             - initialVel: The initial velocity.
             - maxNumberOfParticles: The max number of particles to be emitted.
             - jitter: The jitter amount between 0 and 1.
             - isOneShot: Set true if particles are emitted just once.
             - allowOverlapping: True if particles can be overlapped.
             - seed: The random seed.
             )pbdoc")
        .def_property("surface", &VolumeParticleEmitter3::surface,
                      &VolumeParticleEmitter3::setSurface, R"pbdoc(
             Source surface.
             )pbdoc")
        .def_property("maxRegion", &VolumeParticleEmitter3::maxRegion,
                      &VolumeParticleEmitter3::setMaxRegion, R"pbdoc(
             Max particle gen region.
             )pbdoc")
        .def_property("jitter", &VolumeParticleEmitter3::jitter,
                      &VolumeParticleEmitter3::setJitter, R"pbdoc(
             Jitter amount between 0 and 1.
             )pbdoc")
        .def_property("isOneShot", &VolumeParticleEmitter3::isOneShot,
                      &VolumeParticleEmitter3::setIsOneShot, R"pbdoc(
             True if particles should be emitted just once.
             )pbdoc")
        .def_property("allowOverlapping",
                      &VolumeParticleEmitter3::allowOverlapping,
                      &VolumeParticleEmitter3::setAllowOverlapping, R"pbdoc(
             True if particles can be overlapped.
             )pbdoc")
        .def_property(
            "maxNumberOfParticles", &VolumeParticleEmitter3::maxNumberOfParticles,
            &VolumeParticleEmitter3::setMaxNumberOfParticles, R"pbdoc(
             Max number of particles to be emitted.
             )pbdoc")
        .def_property("spacing", &VolumeParticleEmitter3::spacing,
                      &VolumeParticleEmitter3::setSpacing, R"pbdoc(
             The spacing between particles.
             )pbdoc")
        .def_property(
            "initialVelocity", &VolumeParticleEmitter3::initialVelocity,
            [](VolumeParticleEmitter3& instance, py::object newInitialVel) {
                instance.setInitialVelocity(objectToVector3D(newInitialVel));
            },
            R"pbdoc(
             The initial velocity of the particles.
             )pbdoc")
        .def_property(
            "linearVelocity", &VolumeParticleEmitter3::linearVelocity,
            [](VolumeParticleEmitter3& instance, py::object newLinearVel) {
                instance.setLinearVelocity(objectToVector3D(newLinearVel));
            },
            R"pbdoc(
             The linear velocity of the emitter.
             )pbdoc")
        .def_property(
            "angularVelocity", &VolumeParticleEmitter3::angularVelocity,
            [](VolumeParticleEmitter3& instance, py::object newAngularVel) {
                instance.setAngularVelocity(objectToVector3D(newAngularVel));
            },
            R"pbdoc(
             The angular velocity of the emitter.
             )pbdoc");
}
