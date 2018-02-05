// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array2.h>
#include <jet/implicit_surface_set3.h>
#include <jet/marching_cubes.h>
#include <jet/level_set_liquid_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/plane3.h>
#include <jet/surface_to_implicit3.h>

#include <string>

using namespace jet;

namespace {

void saveTriangleMesh(
    const TriangleMesh3& mesh,
    const std::string& filename) {
    std::ofstream file(filename.c_str());
    if (file) {
        mesh.writeObj(&file);
        file.close();
    }
}

void triangulateAndSave(
    const ScalarGrid3Ptr& sdf,
    const std::string& filename) {
    TriangleMesh3 mesh;
    int flag = kDirectionAll & ~kDirectionDown;
    marchingCubes(
        sdf->constDataAccessor(),
        sdf->gridSpacing(),
        sdf->dataOrigin(),
        &mesh,
        0.0,
        flag);
    saveTriangleMesh(mesh, filename);
}

}  // namespace

JET_TESTS(LevelSetLiquidSolver3);

JET_BEGIN_TEST_F(LevelSetLiquidSolver3, SubtleSloshing) {
    LevelSetLiquidSolver3 solver;

    auto data = solver.gridSystemData();
    double dx = 1.0 / 64.0;
    data->resize({ 64, 64, 8 }, { dx, dx, dx }, Vector3D());

    // Source setting
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Plane3>(
            Vector3D(0.02, 1, 0).normalized(), Vector3D(0.0, 0.5, 0.0)));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector3D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(64, 64);
    auto sdfToBinary = [&](size_t i, size_t j) {
        output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j, 4) / dx);
    };
    output.forEachIndex(sdfToBinary);

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    snprintf(
        filename,
        sizeof(filename),
        "data.#grid2,0000.obj");
    triangulateAndSave(sdf, getFullFilePath(filename));

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; frame.advance()) {
        solver.update(frame);

        output.forEachIndex(sdfToBinary);
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);

        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.obj",
            frame.index);
        triangulateAndSave(sdf, getFullFilePath(filename));
    }
}
JET_END_TEST_F
