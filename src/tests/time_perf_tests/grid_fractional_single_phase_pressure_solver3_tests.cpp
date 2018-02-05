// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid3.h>
#include <jet/face_centered_grid3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>

#include <benchmark/benchmark.h>

using jet::kMaxD;
using jet::Vector3D;
using jet::FaceCenteredGrid3;
using jet::CellCenteredScalarGrid3;
using jet::ConstantScalarField3;
using jet::ConstantVectorField3;

class GridFractionalSinglePhasePressureSolver3 : public ::benchmark::Fixture {
 public:
    FaceCenteredGrid3 vel;
    CellCenteredScalarGrid3 fluidSdf;
    jet::GridFractionalSinglePhasePressureSolver3 solver;

    void SetUp(const ::benchmark::State& state) {
        const auto n = static_cast<size_t>(state.range(0));
        const auto height = static_cast<double>(state.range(1));

        vel.resize(n, n, n);
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

        fluidSdf.resize(n, n, n);
        fluidSdf.fill([&](const Vector3D& x) { return x.y - height; });
    }
};

BENCHMARK_DEFINE_F(GridFractionalSinglePhasePressureSolver3, Solve)
(benchmark::State& state) {
    bool compressed = state.range(2) == 1;
    while (state.KeepRunning()) {
        solver.solve(vel, 1.0, &vel, ConstantScalarField3(kMaxD),
                     ConstantVectorField3({0, 0, 0}), fluidSdf, compressed);
    }
}

BENCHMARK_REGISTER_F(GridFractionalSinglePhasePressureSolver3, Solve)
    ->Args({128, 128, 0})
    ->Args({128, 128, 1})
    ->Args({128, 64, 0})
    ->Args({128, 64, 1})
    ->Args({128, 32, 0})
    ->Args({128, 32, 1});
