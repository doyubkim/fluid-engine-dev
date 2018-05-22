// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "mem_perf_tests.h"

#include <jet/cell_centered_scalar_grid3.h>
#include <jet/face_centered_grid3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>

#include <gtest/gtest.h>

using namespace jet;

namespace {

void runExperiment(size_t n, double height, bool compressed) {
    FaceCenteredGrid3 vel(n, n, n);
    CellCenteredScalarGrid3 fluidSdf(n, n, n);

    vel.fill(Vector3D());

    for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < n + 1; ++j) {
            for (size_t i = 0; i < n; ++i) {
                if (j == 0 || j == n) {
                    vel.v(i, j, k) = 0.0;
                } else {
                    vel.v(i, j, k) = 1.0;
                }
            }
        }
    }

    fluidSdf.fill([&](const Vector3D& x) { return x.y - height * n; });

    GridFractionalSinglePhasePressureSolver3 solver;
    solver.solve(vel, 1.0, &vel, ConstantScalarField3(kMaxD),
                 ConstantVectorField3({0, 0, 0}), fluidSdf, compressed);
}

}  // namespace

TEST(GridFractionalSinglePhasePressureSolver3, FullUncompressed) {
    const size_t mem0 = getCurrentRSS();

    runExperiment(128, 1.0, false);

    const size_t mem1 = getCurrentRSS();

    const auto msg = makeReadableByteSize(mem1 - mem0);

    printMemReport(msg.first, msg.second);
}

TEST(GridFractionalSinglePhasePressureSolver3, FullCompressed) {
    const size_t mem0 = getCurrentRSS();

    runExperiment(128, 1.0, true);

    const size_t mem1 = getCurrentRSS();

    const auto msg = makeReadableByteSize(mem1 - mem0);

    printMemReport(msg.first, msg.second);
}

TEST(GridFractionalSinglePhasePressureSolver3, FreeSurfaceUncompressed) {
    const size_t mem0 = getCurrentRSS();

    runExperiment(128, 0.25, false);

    const size_t mem1 = getCurrentRSS();

    const auto msg = makeReadableByteSize(mem1 - mem0);

    printMemReport(msg.first, msg.second);
}

TEST(GridFractionalSinglePhasePressureSolver3, FreeSurfaceCompressed) {
    const size_t mem0 = getCurrentRSS();

    runExperiment(128, 0.25, true);

    const size_t mem1 = getCurrentRSS();

    const auto msg = makeReadableByteSize(mem1 - mem0);

    printMemReport(msg.first, msg.second);
}
