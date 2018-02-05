// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

JET_TESTS(GridBlockedBoundaryConditionSolver2);

JET_BEGIN_TEST_F(GridBlockedBoundaryConditionSolver2, ConstrainVelocitySmall) {
    GridBlockedBoundaryConditionSolver2 solver;
    auto collider = std::make_shared<RigidBodyCollider2>(
        std::make_shared<Plane2>(Vector2D(1, 1).normalized(), Vector2D()));
    Size2 gridSize(10, 10);
    Vector2D gridSpacing(1.0, 1.0);
    Vector2D gridOrigin(-5.0, -5.0);

    solver.updateCollider(collider, gridSize, gridSpacing, gridOrigin);

    FaceCenteredGrid2 velocity(gridSize, gridSpacing, gridOrigin);
    velocity.fill(Vector2D(1.0, 1.0));

    solver.constrainVelocity(&velocity);

    // Output
    Array2<double> dataU(10, 10);
    Array2<double> dataV(10, 10);
    Array2<double> dataM(10, 10);

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

JET_BEGIN_TEST_F(GridBlockedBoundaryConditionSolver2, ConstrainVelocity) {
    double dx = 1.0 / 32.0;
    FaceCenteredGrid2 velocity(Size2(64, 32), Vector2D(dx, dx), Vector2D());
    velocity.fill(Vector2D(1.0, 0.0));
    BoundingBox2D domain = velocity.boundingBox();

    // Collider setting
    auto surfaceSet = std::make_shared<ImplicitSurfaceSet2>();
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.25));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.upperCorner, 0.25));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.lowerCorner, 0.25));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(surfaceSet);
    collider->linearVelocity = Vector2D(-1.0, 0.0);

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

JET_BEGIN_TEST_F(
    GridBlockedBoundaryConditionSolver2, ConstrainVelocityWithFriction) {
    double dx = 1.0 / 32.0;
    FaceCenteredGrid2 velocity(Size2(64, 32), Vector2D(dx, dx), Vector2D());
    velocity.fill(Vector2D(1.0, 0.0));
    BoundingBox2D domain = velocity.boundingBox();

    // Collider setting
    auto surfaceSet = std::make_shared<ImplicitSurfaceSet2>();
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.25));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.upperCorner, 0.25));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.lowerCorner, 0.25));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(surfaceSet);
    collider->linearVelocity = Vector2D(-1.0, 0.0);
    collider->setFrictionCoefficient(1.0);

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
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.25));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.upperCorner, 0.25));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.lowerCorner, 0.25));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(surfaceSet);
    collider->linearVelocity = Vector2D(-1.0, 0.0);

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
