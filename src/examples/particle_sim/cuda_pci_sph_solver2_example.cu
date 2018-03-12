// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_pci_sph_solver2_example.h"

using namespace jet;
using namespace viz;

namespace {

struct PosToVertex {
    template <typename Tuple>
    JET_CUDA_DEVICE VertexPosition3Color4 operator()(Tuple t) {
        float2 pos = thrust::get<0>(t);
        float d = thrust::get<1>(t);

        d = min(max(d / 1000.0f, 0.0f), 1.0f);
        VertexPosition3Color4 vertex;
        vertex.x = pos.x;
        vertex.y = pos.y;
        vertex.z = 0.0f;
        vertex.r = 1.0f;
        vertex.g = 1.0f - d;
        vertex.b = 1.0f - d;
        vertex.a = 1.0f;
        return vertex;
    }
};

}  // namespace

void CudaPciSphSolver2Example::particlesToVertices() {
    const auto pos = _solver->sphSystemData()->positions();
    const auto den = _solver->sphSystemData()->densities();

    {
        std::lock_guard<std::mutex> lock(_verticesMutex);
        _vertices.resize(pos.size());
        thrust::transform(
            thrust::make_zip_iterator(
                thrust::make_tuple(pos.begin(), den.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(pos.end(), den.end())),
            _vertices.begin(), PosToVertex());
        _areVerticesDirty = true;
    }
}
