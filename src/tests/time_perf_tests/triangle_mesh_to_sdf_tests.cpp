// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/triangle_mesh_to_sdf.h>
#include <jet/vertex_centered_scalar_grid3.h>

#include <benchmark/benchmark.h>

#include <random>

using jet::Vector3D;

class TriangleMeshToSdf : public ::benchmark::Fixture {
 protected:
    std::mt19937 rng{0};
    std::uniform_real_distribution<> dist{0.0, 1.0};
    jet::TriangleMesh3 triMesh;
    jet::VertexCenteredScalarGrid3 grid;

    void SetUp(const ::benchmark::State&) {
        std::ifstream file(RESOURCES_DIR "/bunny.obj");

        if (file) {
            triMesh.readObj(&file);
            file.close();
        }

        jet::BoundingBox3D box = triMesh.boundingBox();
        Vector3D scale(box.width(), box.height(), box.depth());
        box.lowerCorner -= 0.2 * scale;
        box.upperCorner += 0.2 * scale;

        grid.resize(100, 100, 100, box.width() / 100,
             box.height() / 100, box.depth() / 100,
             box.lowerCorner.x, box.lowerCorner.y,
             box.lowerCorner.z);
    }
};

BENCHMARK_DEFINE_F(TriangleMeshToSdf, Call)(benchmark::State& state) {
    while (state.KeepRunning()) {
        triangleMeshToSdf(triMesh, &grid);
    }
}

BENCHMARK_REGISTER_F(TriangleMeshToSdf, Call)->Unit(benchmark::kMillisecond);
