// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/grid_fluid_solver2.h>
#include <jet/grid_single_phase_pressure_solver2.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

JET_TESTS(GridFluidSolver2);

JET_BEGIN_TEST_F(GridFluidSolver2, ApplyBoundaryConditionWithPressure) {
    GridFluidSolver2 solver;
    solver.setGravity(Vector2D(0, 0));
    solver.setAdvectionSolver(nullptr);
    solver.setDiffusionSolver(nullptr);

    auto ppe = std::make_shared<GridSinglePhasePressureSolver2>();
    solver.setPressureSolver(ppe);

    GridSystemData2Ptr data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(64, 32), Vector2D(dx, dx), Vector2D());
    data->velocity()->fill(Vector2D(1.0, 0.0));

    BoundingBox2D domain = data->boundingBox();

    SurfaceToImplicit2Ptr sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(domain.midPoint(), 0.25));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(sphere);
    solver.setCollider(collider);

    Frame frame(0, 1.0 / 60.0);
    solver.update(frame);

    Array2<double> dataU(64, 32);
    Array2<double> dataV(64, 32);
    Array2<double> div(64, 32);
    Array2<double> pressure(64, 32);

    dataU.forEachIndex([&](size_t i, size_t j) {
        Vector2D vel = data->velocity()->valueAtCellCenter(i, j);
        dataU(i, j) = vel.x;
        dataV(i, j) = vel.y;
        div(i, j) = data->velocity()->divergenceAtCellCenter(i, j);
        pressure(i, j) = ppe->pressure()(i, j);
    });

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
    saveData(div.constAccessor(), "div_#grid2.npy");
    saveData(pressure.constAccessor(), "pressure_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(
    GridFluidSolver2,
    ApplyBoundaryConditionWithVariationalPressure) {
    GridFluidSolver2 solver;
    solver.setGravity(Vector2D(0, 0));
    solver.setAdvectionSolver(nullptr);
    solver.setDiffusionSolver(nullptr);

    auto ppe = std::make_shared<GridFractionalSinglePhasePressureSolver2>();
    solver.setPressureSolver(ppe);

    GridSystemData2Ptr data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(64, 32), Vector2D(dx, dx), Vector2D());
    data->velocity()->fill(Vector2D(1.0, 0.0));

    BoundingBox2D domain = data->boundingBox();

    SurfaceToImplicit2Ptr sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(domain.midPoint(), 0.25));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(sphere);
    solver.setCollider(collider);

    Frame frame(0, 1.0 / 60.0);
    solver.update(frame);

    Array2<double> dataU(64, 32);
    Array2<double> dataV(64, 32);
    Array2<double> div(64, 32);
    Array2<double> pressure(64, 32);

    dataU.forEachIndex([&](size_t i, size_t j) {
        Vector2D vel = data->velocity()->valueAtCellCenter(i, j);
        dataU(i, j) = vel.x;
        dataV(i, j) = vel.y;
        div(i, j) = data->velocity()->divergenceAtCellCenter(i, j);
        pressure(i, j) = ppe->pressure()(i, j);
    });

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
    saveData(div.constAccessor(), "div_#grid2.npy");
    saveData(pressure.constAccessor(), "pressure_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridFluidSolver2, ApplyBoundaryConditionWithPressureOpen) {
    GridFluidSolver2 solver;
    solver.setGravity(Vector2D(0, 0));
    solver.setAdvectionSolver(nullptr);
    solver.setDiffusionSolver(nullptr);

    // Open left and right
    solver.setClosedDomainBoundaryFlag(kDirectionDown | kDirectionUp);

    GridSinglePhasePressureSolver2Ptr ppe
        = std::make_shared<GridSinglePhasePressureSolver2>();
    solver.setPressureSolver(ppe);

    GridSystemData2Ptr data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(64, 32), Vector2D(dx, dx), Vector2D());
    data->velocity()->fill(Vector2D(1.0, 0.0));

    BoundingBox2D domain = data->boundingBox();

    SurfaceToImplicit2Ptr sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(domain.midPoint(), 0.25));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(sphere);
    solver.setCollider(collider);

    Frame frame(0, 1.0 / 60.0);
    solver.update(frame);

    Array2<double> dataU(64, 32);
    Array2<double> dataV(64, 32);
    Array2<double> div(64, 32);
    Array2<double> pressure(64, 32);

    dataU.forEachIndex([&](size_t i, size_t j) {
        Vector2D vel = data->velocity()->valueAtCellCenter(i, j);
        dataU(i, j) = vel.x;
        dataV(i, j) = vel.y;
        div(i, j) = data->velocity()->divergenceAtCellCenter(i, j);
        pressure(i, j) = ppe->pressure()(i, j);
    });

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
    saveData(div.constAccessor(), "div_#grid2.npy");
    saveData(pressure.constAccessor(), "pressure_#grid2.npy");
}
JET_END_TEST_F
