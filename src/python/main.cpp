// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "animation.h"
#include "bounding_box.h"
#include "collider.h"
#include "flip_solver.h"
#include "grid_fluid_solver.h"
#include "implicit_surface.h"
#include "logging.h"
#include "particle_emitter.h"
#include "particle_system_data.h"
#include "physics_animation.h"
#include "pic_solver.h"
#include "quaternion.h"
#include "ray.h"
#include "rigid_body_collider.h"
#include "size.h"
#include "sphere.h"
#include "surface.h"
#include "transform.h"
#include "vector.h"
#include "volume_particle_emitter.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_PLUGIN(pyjet) {
    py::module m("pyjet",
                 "Fluid simulation engine for computer graphics applications");

    // Trivial basic types
    addBoundingBox3D(m);
    addBoundingBox3F(m);
    addFrame(m);
    addQuaternionD(m);
    addQuaternionF(m);
    addRay3D(m);
    addRay3F(m);
    addSize2(m);
    addSize3(m);
    addTransform3(m);
    addVector3D(m);
    addVector3F(m);

    // Trivial APIs
    addLogging(m);

    // Animations
    addAnimation(m);
    addPhysicsAnimation(m);
    addGridFluidSolver3(m);
    addPicSolver3(m);
    addFlipSolver3(m);

    // Colliders
    addCollider3(m);
    addRigidBodyCollider3(m);

    // Particle emitters
    addParticleEmitter3(m);
    addVolumeParticleEmitter3(m);

    // Particle systems
    addParticleSystemData3(m);

    // Surfaces
    addSurface3(m);
    addSphere3(m);
    addImplicitSurface3(m);

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}