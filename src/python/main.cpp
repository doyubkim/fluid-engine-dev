// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "advection_solver.h"
#include "animation.h"
#include "apic_solver.h"
#include "array_accessor1.h"
#include "bounding_box.h"
#include "cell_centered_scalar_grid.h"
#include "cell_centered_vector_grid.h"
#include "collider.h"
#include "collocated_vector_grid.h"
#include "constants.h"
#include "face_centered_grid.h"
#include "field.h"
#include "flip_solver.h"
#include "grid.h"
#include "grid_backward_euler_diffusion_solver.h"
#include "grid_diffusion_solver.h"
#include "grid_emitter.h"
#include "grid_fluid_solver.h"
#include "grid_forward_euler_diffusion_solver.h"
#include "grid_pressure_solver.h"
#include "grid_system_data.h"
#include "implicit_surface.h"
#include "logging.h"
#include "particle_emitter.h"
#include "particle_system_data.h"
#include "physics_animation.h"
#include "pic_solver.h"
#include "quaternion.h"
#include "ray.h"
#include "rigid_body_collider.h"
#include "scalar_field.h"
#include "scalar_grid.h"
#include "size.h"
#include "sphere.h"
#include "surface.h"
#include "transform.h"
#include "vector.h"
#include "vector_field.h"
#include "vector_grid.h"
#include "vertex_centered_scalar_grid.h"
#include "vertex_centered_vector_grid.h"
#include "volume_particle_emitter.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_PLUGIN(pyjet) {
    py::module m("pyjet", R"pbdoc(
        Fluid simulation engine for computer graphics applications
    )pbdoc");

    // Constants
    addConstants(m);

    // Trivial basic types
    addArrayAccessor1(m);
    addVector2D(m);
    addVector2F(m);
    addVector3D(m);
    addVector3F(m);
    addRay2D(m);
    addRay2F(m);
    addRay3D(m);
    addRay3F(m);
    addBoundingBox2D(m);
    addBoundingBox2F(m);
    addBoundingBox3D(m);
    addBoundingBox3F(m);
    addFrame(m);
    addQuaternionD(m);
    addQuaternionF(m);
    addSize2(m);
    addSize3(m);
    addTransform3(m);

    // Trivial APIs
    addLogging(m);

    // Fields/Grids
    addField2(m);
    addField3(m);
    addScalarField2(m);
    addScalarField3(m);
    addVectorField2(m);
    addVectorField3(m);
    addGrid2(m);
    addGrid3(m);
    addScalarGrid2(m);
    addScalarGrid3(m);
    addVectorGrid2(m);
    addVectorGrid3(m);
    addCollocatedVectorGrid2(m);
    addCollocatedVectorGrid3(m);
    addCellCenteredScalarGrid2(m);
    addCellCenteredScalarGrid3(m);
    addCellCenteredVectorGrid2(m);
    addCellCenteredVectorGrid3(m);
    addVertexCenteredScalarGrid2(m);
    addVertexCenteredScalarGrid3(m);
    addVertexCenteredVectorGrid2(m);
    addVertexCenteredVectorGrid3(m);
    addFaceCenteredGrid2(m);
    addFaceCenteredGrid3(m);

    // Surfaces
    addSurface2(m);
    addSurface3(m);
    addSphere2(m);
    addSphere3(m);
    addImplicitSurface3(m);

    // Data models
    addGridSystemData2(m);
    addGridSystemData3(m);
    addParticleSystemData2(m);
    addParticleSystemData3(m);

    // Emitters
    addGridEmitter2(m);
    addGridEmitter3(m);
    addParticleEmitter2(m);
    addParticleEmitter3(m);
    addVolumeParticleEmitter2(m);
    addVolumeParticleEmitter3(m);

    // Colliders
    addCollider2(m);
    addCollider3(m);
    addRigidBodyCollider2(m);
    addRigidBodyCollider3(m);

    // Solvers
    addAdvectionSolver2(m);
    addAdvectionSolver3(m);
    addGridDiffusionSolver2(m);
    addGridDiffusionSolver3(m);
    addGridForwardEulerDiffusionSolver2(m);
    addGridForwardEulerDiffusionSolver3(m);
    addGridBackwardEulerDiffusionSolver2(m);
    addGridBackwardEulerDiffusionSolver3(m);
    addGridPressureSolver2(m);
    addGridPressureSolver3(m);

    // Animations
    addAnimation(m);
    addPhysicsAnimation(m);
    addGridFluidSolver2(m);
    addGridFluidSolver3(m);
    addPicSolver2(m);
    addPicSolver3(m);
    addFlipSolver2(m);
    addFlipSolver3(m);
    addApicSolver2(m);
    addApicSolver3(m);

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}