// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_particle_system_solver3_example.h"

#include <jet.viz/points_renderable3.h>

#include <thrust/random.h>

using namespace jet;
using namespace experimental;
using namespace viz;

namespace {

struct Rng {
    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        size_t idx = thrust::get<0>(t);

        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist(0.0f, 1.0f);

        float4 result;
        randEng.discard(3 * idx);
        result.x = uniDist(randEng);
        randEng.discard(3 * idx + 1);
        result.y = uniDist(randEng);
        randEng.discard(3 * idx + 2);
        result.z = uniDist(randEng);
        result.w = 0.0f;

        thrust::get<1>(t) = result;
    }
};

struct PosToVertex {
    template <typename Pos>
    __device__ VertexPosition3Color4 operator()(const Pos& pos) {
        VertexPosition3Color4 vertex;
        vertex.x = pos.x;
        vertex.y = pos.y;
        vertex.z = pos.z;
        vertex.r = 1.0f;
        vertex.g = 1.0f;
        vertex.b = 1.0f;
        vertex.a = 1.0f;
        return vertex;
    }
};

}  // namespace

CudaParticleSystemSolver3Example::CudaParticleSystemSolver3Example(
    const jet::Frame& frame)
    : ParticleSimExample(frame) {}

void CudaParticleSystemSolver3Example::onSetup(GlfwWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>(Vector3D{0.5, 0.5, 3.0},
                                      Vector3D{0.0, 0.0, -1.0},
                                      Vector3D{0.0, 1.0, 0.0}, 0.01, 10.0),
        Vector3D{0.5, 0.5, 0.5}));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{0, 0, 0, 1});

    // Setup solver
    _solver =
        jet::experimental::CudaParticleSystemSolver3::builder().makeShared();
    _solver->setDragCoefficient(0.0);
    _solver->setRestitutionCoefficient(1.0);

    size_t numParticles = static_cast<size_t>(1 << 14);
    CudaParticleSystemData3* particles = _solver->particleSystemData();

    thrust::device_vector<float4> pos(numParticles);
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(jet::kZeroSize), pos.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(numParticles), pos.end())),
        Rng());
    particles->addParticles(jet::experimental::CudaArrayView1<float4>(pos));

    // Setup renderable
    thrust::device_vector<VertexPosition3Color4> vertices(numParticles);
    thrust::transform(pos.begin(), pos.end(), vertices.begin(), PosToVertex());

    _renderable = std::make_shared<PointsRenderable3>(renderer);
    _renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                           window->windowSize().x);
    _renderable->setPositionsAndColors(nullptr, vertices.size());
    _renderable->vertexBuffer()->updateWithCuda(
        (const float*)thrust::raw_pointer_cast(vertices.data()));
    renderer->addRenderable(_renderable);
}

void CudaParticleSystemSolver3Example::onUpdate(const Frame& frame) {
    _solver->update(frame);

    auto pos = _solver->particleSystemData()->positions();

    thrust::device_ptr<VertexPosition3Color4> vertices(
        (VertexPosition3Color4*)_renderable->vertexBuffer()->cudaMapResources());
    thrust::transform(pos.begin(), pos.end(), vertices, PosToVertex());
    _renderable->vertexBuffer()->cudaUnmapResources();
}
