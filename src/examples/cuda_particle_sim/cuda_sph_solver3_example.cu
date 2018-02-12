// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_sph_solver3_example.h"

#include <jet.viz/points_renderable3.h>
#include <jet/cuda_array1.h>
#include <jet/grid_point_generator3.h>

#include <thrust/random.h>

using namespace jet;
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
    template <typename Tuple>
    __device__ VertexPosition3Color4 operator()(Tuple t) {
        float4 pos = thrust::get<0>(t);
        float d = thrust::get<1>(t);

        d = min(max(d / 1000.0f, 0.0f), 1.0f);
        VertexPosition3Color4 vertex;
        vertex.x = pos.x;
        vertex.y = pos.y;
        vertex.z = pos.z;
        vertex.r = 1.0f;
        vertex.g = 1.0f - d;
        vertex.b = 1.0f - d;
        vertex.a = 1.0f;
        return vertex;
    }
};

}  // namespace

CudaSphSolver3Example::CudaSphSolver3Example(const jet::Frame& frame)
    : ParticleSimExample(frame) {}

void CudaSphSolver3Example::onSetup(GlfwWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>(Vector3D{0.5, 0.5, 2.0},
                                      Vector3D{0.0, 0.0, -1.0},
                                      Vector3D{0.0, 1.0, 0.0}, 0.01, 10.0),
        Vector3D{0.5, 0.5, 0.5}));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{0, 0, 0, 1});

    // Setup solver
    const float targetSpacing = 1.0f / 50.0f;
    _solver = jet::experimental::CudaWcSphSolver3::builder().makeShared();
    _solver->setDragCoefficient(0.0f);
    _solver->setRestitutionCoefficient(1.0f);
    _solver->setViscosityCoefficient(0.1f);
    _solver->setPseudoViscosityCoefficient(10.0f);
    _solver->setIsUsingFixedSubTimeSteps(true);
    _solver->setNumberOfFixedSubTimeSteps(1);

    auto particles = _solver->sphSystemData();
    particles->setTargetSpacing(targetSpacing);
    particles->setRelativeKernelRadius(1.8f);

    // Seed particles
    BoundingBox3D vol(Vector3D(), Vector3D(0.5, 0.5, 0.5));
    vol.expand(-targetSpacing);
    Array1<Vector3D> rawPoints;
    GridPointGenerator3 generator;
    generator.generate(vol, targetSpacing, &rawPoints);
    Array1<float4> hostData(rawPoints.size());
    for (size_t i = 0; i < rawPoints.size(); ++i) {
        auto rp = rawPoints[i].castTo<float>();
        hostData[i] = make_float4(rp.x, rp.y, rp.z, 0.0f);
    }
    experimental::CudaArray1<float4> deviceData(hostData);
    particles->addParticles(deviceData);
    printf("%zu particles generated.\n", deviceData.size());

    // Setup renderable
    thrust::device_vector<VertexPosition3Color4> vertices(deviceData.size());
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          deviceData.begin(), particles->densities().begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          deviceData.end(), particles->densities().end())),
                      vertices.begin(), PosToVertex());

    _renderable = std::make_shared<PointsRenderable3>(renderer);
    _renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                           window->windowSize().x);
    _renderable->setPositionsAndColors(nullptr, vertices.size());
    _renderable->vertexBuffer()->updateWithCuda(
        (const float*)thrust::raw_pointer_cast(vertices.data()));
    renderer->addRenderable(_renderable);
}

void CudaSphSolver3Example::onUpdate(const Frame& frame) {
    _solver->update(frame);

    auto pos = _solver->sphSystemData()->positions();
    auto den = _solver->sphSystemData()->densities();

    thrust::device_ptr<VertexPosition3Color4> vertices(
        (VertexPosition3Color4*)_renderable->vertexBuffer()
            ->cudaMapResources());
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(pos.begin(), den.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(pos.end(), den.end())),
        vertices, PosToVertex());
    _renderable->vertexBuffer()->cudaUnmapResources();
}
