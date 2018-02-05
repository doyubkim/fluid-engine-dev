// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/particle_system_solver2.h>
#include <jet/particle_system_solver3.h>
#include <jet/sphere2.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit2.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter2.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

JET_TESTS(VolumeParticleEmitter2);

JET_BEGIN_TEST_F(VolumeParticleEmitter2, EmitContinuousNonOverlapping) {
    ParticleSystemSolver2 solver;

    ParticleSystemData2Ptr particles = solver.particleSystemData();
    auto emitter = std::make_shared<VolumeParticleEmitter2>(
        std::make_shared<SurfaceToImplicit2>(
            std::make_shared<Sphere2>(Vector2D(), 1.0)),
        BoundingBox2D(Vector2D(-1, -1), Vector2D(1, 1)),
        0.2);
    emitter->setIsOneShot(false);
    emitter->setAllowOverlapping(false);
    solver.setEmitter(emitter);

    saveParticleDataXy(particles, 0);

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; ++frame) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F

JET_TESTS(VolumeParticleEmitter3);

JET_BEGIN_TEST_F(VolumeParticleEmitter3, EmitContinuousNonOverlapping) {
    ParticleSystemSolver3 solver;

    ParticleSystemData3Ptr particles = solver.particleSystemData();
    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        std::make_shared<SurfaceToImplicit3>(
            std::make_shared<Sphere3>(Vector3D(), 1.0)),
        BoundingBox3D(Vector3D(-1, -1, -1), Vector3D(1, 1, 1)),
        0.2);
    emitter->setIsOneShot(false);
    emitter->setAllowOverlapping(false);
    solver.setEmitter(emitter);

    saveParticleDataXy(particles, 0);

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; ++frame) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F
