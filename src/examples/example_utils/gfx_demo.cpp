// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "gfx_demo.h"
#include "gfx_example_manager.h"

using namespace jet;
using namespace gfx;

class SimpleExample final : public GfxExample {
 public:
    SimpleExample() : GfxExample(Frame()) {}

    std::string name() const override { return "Simple Example"; }

    void onSetup(Window* window) override {
        window->renderer()->setBackgroundColor({1.0f, 0.5f, 0.2f, 1.0f});
    }
};

class PointsExample final : public GfxExample {
 public:
    PointsExample() : GfxExample(Frame()) {}

    std::string name() const override { return "Points Example"; }

    void onSetup(Window* window) override {
        Array1<Vector3F> positions;
        Array1<Vector4F> colors;

        // Generate random points
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist1(-0.5f, 0.5f);
        std::uniform_real_distribution<float> dist2(-1.0f, 1.0f);
        for (size_t i = 0; i < 1000; ++i) {
            positions.append({dist1(rng), dist1(rng), dist1(rng)});
            colors.append(ColorUtils::makeJet(dist2(rng)));
        }

        auto pointsRenderable = std::make_shared<PointsRenderable>(
            positions, colors, 10.0f * window->displayScalingFactor().x);
        window->renderer()->addRenderable(pointsRenderable);
        window->renderer()->setBackgroundColor({0.2f, 0.5f, 1.0f, 1.0f});

        Viewport viewport(0, 0, window->framebufferSize().x,
                          window->framebufferSize().y);
        CameraState camera{.origin = Vector3F(0, 0, 1),
                           .lookAt = Vector3F(0, 0, -1),
                           .viewport = viewport};
        window->setViewController(std::make_shared<PitchYawViewController>(
            std::make_shared<PerspCamera>(camera, kHalfPiF), Vector3F()));
    }
};

class SimpleParticleAnimationExample final : public GfxExample {
 public:
    SimpleParticleAnimationExample() : GfxExample(Frame()) {}

    std::string name() const override {
        return "Simple Particle Animation Example";
    }

    void onSetup(Window* window) override {
        _window = window;

        // Set up sim
        Plane2Ptr plane = std::make_shared<Plane2>(Vector2D(0, 1), Vector2D());
        RigidBodyCollider2Ptr collider =
            std::make_shared<RigidBodyCollider2>(plane);
        collider->setFrictionCoefficient(0.01);
        _solver.setRestitutionCoefficient(0.5);
        _solver.setCollider(collider);

        ParticleSystemData2Ptr particles = _solver.particleSystemData();
        PointParticleEmitter2Ptr emitter =
            std::make_shared<PointParticleEmitter2>(Vector2D(0, 0),
                                                    Vector2D(0, 1), 10.0, 15.0);
        emitter->setMaxNumberOfNewParticlesPerSecond(100);
        emitter->setMaxNumberOfParticles(1000);
        _solver.setEmitter(emitter);

        // Set up rendering
        Array1<Vector3F> positions(1000, Vector3F());
        Array1<Vector4F> colors(1000, Vector4F());
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist2(-1.0f, 1.0f);
        for (size_t i = 0; i < 1000; ++i) {
            positions[i] = Vector3F(0, -1, 0);  // hide outside the screen :)
            colors[i] = ColorUtils::makeJet(dist2(rng));
        }

        _renderable = std::make_shared<PointsRenderable>(
            positions, colors, _radius * window->displayScalingFactor().x);
        window->renderer()->addRenderable(_renderable);
        window->renderer()->setBackgroundColor({0.1f, 0.1f, 0.1f, 1.0f});

        window->setSwapInterval(1);
        Viewport viewport(0, 0, window->framebufferSize().x,
                          window->framebufferSize().y);
        CameraState camera{.origin = Vector3F(0, 0, 1),
                           .lookAt = Vector3F(0, 0, -1),
                           .viewport = viewport};
        window->setViewController(std::make_shared<OrthoViewController>(
            std::make_shared<OrthoCamera>(camera, -3, 3, 0, 6)));
    }

    void onAdvanceSim(const jet::Frame& frame) override {
        _solver.update(frame);

        ParticleSystemData2Ptr particles = _solver.particleSystemData();
        auto particlePosView = particles->positions();
        Array1<Vector3F> positions(1000, Vector3F());

        for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
            Vector3F pt((float)particlePosView[i].x,
                        (float)particlePosView[i].y, 0);
            positions[i] = pt;
        }

        _renderable->update(positions, Array1<Vector4F>(),
                            _radius * _window->displayScalingFactor().x);
    }

 private:
    ParticleSystemSolver2 _solver;
    PointsRenderablePtr _renderable;
    Window* _window;

    float _radius = 4.0;
};

void makeGfxDemo(const WindowPtr& window) {
    Logging::mute();
    GfxExampleManager::initialize(window);
    GfxExampleManager::addExample<SimpleExample>();
    GfxExampleManager::addExample<PointsExample>();
    GfxExampleManager::addExample<SimpleParticleAnimationExample>();
}
