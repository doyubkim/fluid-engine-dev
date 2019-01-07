// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "advection_solver.h"
#include "animation.h"
#include "anisotropic_points_to_implicit.h"
#include "apic_solver.h"
#include "array_accessor.h"
#include "bounding_box.h"
#include "box.h"
#include "cell_centered_scalar_grid.h"
#include "cell_centered_vector_grid.h"
#include "collider.h"
#include "collider_set.h"
#include "collocated_vector_grid.h"
#include "constant_scalar_field.h"
#include "constant_vector_field.h"
#include "constants.h"
#include "cubic_semi_lagrangian.h"
#include "custom_scalar_field.h"
#include "custom_vector_field.h"
#include "cylinder.h"
#include "eno_level_set_solver.h"
#include "face_centered_grid.h"
#include "fdm_cg_solver.h"
#include "fdm_gauss_seidel_solver.h"
#include "fdm_iccg_solver.h"
#include "fdm_jacobi_solver.h"
#include "fdm_linear_system_solver.h"
#include "fdm_mg_solver.h"
#include "fdm_mgpcg_solver.h"
#include "field.h"
#include "flip_solver.h"
#include "fmm_level_set_solver.h"
#include "grid.h"
#include "grid_backward_euler_diffusion_solver.h"
#include "grid_blocked_boundary_condition_solver.h"
#include "grid_boundary_condition_solver.h"
#include "grid_diffusion_solver.h"
#include "grid_emitter.h"
#include "grid_fluid_solver.h"
#include "grid_forward_euler_diffusion_solver.h"
#include "grid_fractional_boundary_condition_solver.h"
#include "grid_fractional_single_phase_pressure_solver.h"
#include "grid_pressure_solver.h"
#include "grid_single_phase_pressure_solver.h"
#include "grid_smoke_solver.h"
#include "grid_system_data.h"
#include "implicit_surface.h"
#include "implicit_triangle_mesh.h"
#include "iterative_level_set_solver.h"
#include "level_set_liquid_solver.h"
#include "level_set_solver.h"
#include "logging.h"
#include "marching_cubes.h"
#include "particle_emitter.h"
#include "particle_emitter_set.h"
#include "particle_system_data.h"
#include "particle_system_solver.h"
#include "pci_sph_solver.h"
#include "physics_animation.h"
#include "pic_solver.h"
#include "plane.h"
#include "point.h"
#include "point_particle_emitter.h"
#include "points_to_implicit.h"
#include "quaternion.h"
#include "ray.h"
#include "rigid_body_collider.h"
#include "scalar_field.h"
#include "scalar_grid.h"
#include "semi_lagrangian.h"
#include "serializable.h"
#include "size.h"
#include "sph_points_to_implicit.h"
#include "sph_solver.h"
#include "sph_system_data.h"
#include "sphere.h"
#include "spherical_points_to_implicit.h"
#include "surface.h"
#include "surface_set.h"
#include "surface_to_implicit.h"
#include "transform.h"
#include "triangle.h"
#include "triangle_mesh.h"
#include "upwind_level_set_solver.h"
#include "vector.h"
#include "vector_field.h"
#include "vector_grid.h"
#include "vertex_centered_scalar_grid.h"
#include "vertex_centered_vector_grid.h"
#include "volume_grid_emitter.h"
#include "volume_particle_emitter.h"
#include "zhu_bridson_points_to_implicit.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pyjet, m) {
    m.doc() = "Fluid simulation engine for computer graphics applications";

    // Constants
    addConstants(m);

    // Serialization
    addSerializable(m);

    // Trivial basic types
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
    addPoint2UI(m);
    addPoint3UI(m);
    addSize2(m);
    addSize3(m);
    addTransform2(m);
    addTransform3(m);

    // Containers/helpers
    addArrayAccessor1(m);
    addArrayAccessor2(m);
    addArrayAccessor3(m);

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
    addConstantScalarField2(m);
    addConstantScalarField3(m);
    addConstantVectorField2(m);
    addConstantVectorField3(m);
    addCustomScalarField2(m);
    addCustomScalarField3(m);
    addCustomVectorField2(m);
    addCustomVectorField3(m);

    // Surfaces
    addSurface2(m);
    addSurface3(m);
    addSurfaceSet2(m);
    addSurfaceSet3(m);
    addCylinder3(m);
    addBox2(m);
    addBox3(m);
    addPlane2(m);
    addPlane3(m);
    addSphere2(m);
    addSphere3(m);
    addTriangle3(m);
    addTriangleMesh3(m);
    addImplicitSurface2(m);
    addImplicitSurface3(m);
    addSurfaceToImplicit2(m);
    addSurfaceToImplicit3(m);
    addImplicitTriangleMesh3(m);

    // Data models
    addGridSystemData2(m);
    addGridSystemData3(m);

    addParticleSystemData2(m);
    addParticleSystemData3(m);
    addSphSystemData2(m);
    addSphSystemData3(m);

    // Emitters
    addGridEmitter2(m);
    addGridEmitter3(m);
    addVolumeGridEmitter2(m);
    addVolumeGridEmitter3(m);
    addParticleEmitter2(m);
    addParticleEmitter3(m);
    addParticleEmitterSet2(m);
    addParticleEmitterSet3(m);
    addPointParticleEmitter2(m);
    addPointParticleEmitter3(m);
    addVolumeParticleEmitter2(m);
    addVolumeParticleEmitter3(m);

    // Colliders
    addCollider2(m);
    addCollider3(m);
    addColliderSet2(m);
    addColliderSet3(m);
    addRigidBodyCollider2(m);
    addRigidBodyCollider3(m);

    // Solvers
    addAdvectionSolver2(m);
    addAdvectionSolver3(m);
    addSemiLagrangian2(m);
    addSemiLagrangian3(m);
    addCubicSemiLagrangian2(m);
    addCubicSemiLagrangian3(m);
    addFdmLinearSystemSolver2(m);
    addFdmLinearSystemSolver3(m);
    addFdmJacobiSolver2(m);
    addFdmJacobiSolver3(m);
    addFdmGaussSeidelSolver2(m);
    addFdmGaussSeidelSolver3(m);
    addFdmCgSolver2(m);
    addFdmCgSolver3(m);
    addFdmIccgSolver2(m);
    addFdmIccgSolver3(m);
    addFdmMgSolver2(m);
    addFdmMgSolver3(m);
    addFdmMgpcgSolver2(m);
    addFdmMgpcgSolver3(m);
    addGridDiffusionSolver2(m);
    addGridDiffusionSolver3(m);
    addGridForwardEulerDiffusionSolver2(m);
    addGridForwardEulerDiffusionSolver3(m);
    addGridBackwardEulerDiffusionSolver2(m);
    addGridBackwardEulerDiffusionSolver3(m);
    addGridBoundaryConditionSolver2(m);
    addGridBoundaryConditionSolver3(m);
    addGridFractionalBoundaryConditionSolver2(m);
    addGridFractionalBoundaryConditionSolver3(m);
    addGridBlockedBoundaryConditionSolver2(m);
    addGridBlockedBoundaryConditionSolver3(m);
    addGridPressureSolver2(m);
    addGridPressureSolver3(m);
    addGridSinglePhasePressureSolver2(m);
    addGridSinglePhasePressureSolver3(m);
    addGridFractionalSinglePhasePressureSolver2(m);
    addGridFractionalSinglePhasePressureSolver3(m);
    addLevelSetSolver2(m);
    addLevelSetSolver3(m);
    addIterativeLevelSetSolver2(m);
    addIterativeLevelSetSolver3(m);
    addUpwindLevelSetSolver2(m);
    addUpwindLevelSetSolver3(m);
    addEnoLevelSetSolver2(m);
    addEnoLevelSetSolver3(m);
    addFmmLevelSetSolver2(m);
    addFmmLevelSetSolver3(m);
    addPointsToImplicit2(m);
    addPointsToImplicit3(m);
    addSphericalPointsToImplicit2(m);
    addSphericalPointsToImplicit3(m);
    addSphPointsToImplicit2(m);
    addSphPointsToImplicit3(m);
    addZhuBridsonPointsToImplicit2(m);
    addZhuBridsonPointsToImplicit3(m);
    addAnisotropicPointsToImplicit2(m);
    addAnisotropicPointsToImplicit3(m);

    // Animations
    addAnimation(m);
    addPhysicsAnimation(m);
    addGridFluidSolver2(m);
    addGridFluidSolver3(m);
    addGridSmokeSolver2(m);
    addGridSmokeSolver3(m);
    addLevelSetLiquidSolver2(m);
    addLevelSetLiquidSolver3(m);
    addPicSolver2(m);
    addPicSolver3(m);
    addFlipSolver2(m);
    addFlipSolver3(m);
    addApicSolver2(m);
    addApicSolver3(m);
    addParticleSystemSolver2(m);
    addParticleSystemSolver3(m);
    addSphSolver2(m);
    addSphSolver3(m);
    addPciSphSolver2(m);
    addPciSphSolver3(m);

    // Global functions
    addMarchingCubes(m);

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
}
