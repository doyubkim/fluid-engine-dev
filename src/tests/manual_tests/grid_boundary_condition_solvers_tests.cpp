// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

JET_TESTS(GridBlockedBoundaryConditionSolver2);

JET_BEGIN_TEST_F(GridBlockedBoundaryConditionSolver2, ConstrainVelocity) {
    double dx = 1.0 / 32.0;
    FaceCenteredGrid2 velocity(Size2(64, 32), Vector2D(dx, dx), Vector2D());
    velocity.fill(Vector2D(1.0, 0.0));
    BoundingBox2D domain = velocity.boundingBox();

    // Collider setting
    auto surfaceSet = std::make_shared<ImplicitSurfaceSet2>();
    surfaceSet->addSurface(std::make_shared<Sphere2>(domain.midPoint(), 0.25));
    surfaceSet->addSurface(std::make_shared<Sphere2>(domain.upperCorner, 0.25));
    surfaceSet->addSurface(std::make_shared<Sphere2>(domain.lowerCorner, 0.25));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(surfaceSet);
    collider->setLinearVelocity(Vector2D(-1.0, 0.0));

    // Solver setting
    GridBlockedBoundaryConditionSolver2 solver;
    solver.updateCollider(
        collider,
        velocity.resolution(),
        velocity.gridSpacing(),
        velocity.origin());
    solver.setClosedDomainBoundaryFlag(
        kDirectionRight | kDirectionDown | kDirectionUp);

    // Constrain velocity
    solver.constrainVelocity(&velocity, 5);

    // Output
    Array2<double> dataU(64, 32);
    Array2<double> dataV(64, 32);
    Array2<double> dataM(64, 32);

    dataU.forEachIndex([&](size_t i, size_t j) {
        Vector2D vel = velocity.valueAtCellCenter(i, j);
        dataU(i, j) = vel.x;
        dataV(i, j) = vel.y;
        dataM(i, j) = static_cast<double>(solver.marker()(i, j));
    });

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
    saveData(dataM.constAccessor(), "marker_#grid2.npy");
}
JET_END_TEST_F

JET_TESTS(GridFractionalBoundaryConditionSolver2);

JET_BEGIN_TEST_F(
    GridFractionalBoundaryConditionSolver2,
    ConstrainVelocity) {
    double dx = 1.0 / 32.0;
    FaceCenteredGrid2 velocity(Size2(64, 32), Vector2D(dx, dx), Vector2D());
    velocity.fill(Vector2D(1.0, 0.0));
    BoundingBox2D domain = velocity.boundingBox();

    // Collider setting
    auto surfaceSet = std::make_shared<ImplicitSurfaceSet2>();
    surfaceSet->addSurface(std::make_shared<Sphere2>(domain.midPoint(), 0.25));
    surfaceSet->addSurface(std::make_shared<Sphere2>(domain.upperCorner, 0.25));
    surfaceSet->addSurface(std::make_shared<Sphere2>(domain.lowerCorner, 0.25));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(surfaceSet);
    collider->setLinearVelocity(Vector2D(-1.0, 0.0));

    // Solver setting
    GridFractionalBoundaryConditionSolver2 solver;
    solver.updateCollider(
        collider,
        velocity.resolution(),
        velocity.gridSpacing(),
        velocity.origin());
    solver.setClosedDomainBoundaryFlag(
        kDirectionRight | kDirectionDown | kDirectionUp);

    // Constrain velocity
    solver.constrainVelocity(&velocity, 5);

    // Output
    Array2<double> dataU(64, 32);
    Array2<double> dataV(64, 32);

    dataU.forEachIndex([&](size_t i, size_t j) {
        Vector2D vel = velocity.valueAtCellCenter(i, j);
        dataU(i, j) = vel.x;
        dataV(i, j) = vel.y;
    });

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
}
JET_END_TEST_F
