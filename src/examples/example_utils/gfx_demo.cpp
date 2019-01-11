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
    }
};

class SimpleParticleAnimationExample final : public GfxExample {
 public:
    SimpleParticleAnimationExample() : GfxExample(Frame()) {}

    std::string name() const override {
        return "Simple Particle Animation Example";
    }

    void onSetup(Window* window) override {
        // Set up sim
        onResetSim();

        // Set up rendering
        _renderable = std::make_shared<PointsRenderable>(
            Array1<Vector3F>(), Array1<Vector4F>(),
            4.0f * window->displayScalingFactor().x);
        window->renderer()->addRenderable(_renderable);
        window->renderer()->setBackgroundColor({0.1f, 0.1f, 0.1f, 1.0f});
        window->setSwapInterval(1);
    }

    void onAdvanceSim(const jet::Frame& frame) override {
        _solver.update(frame);
    }

 private:
    ParticleSystemSolver2 _solver;
    Array1<Vector3F> _positions;
    Array1<Vector4F> _colors;
    PointsRenderablePtr _renderable;
    std::mt19937 _rng{0};

    void onResetView(Window* window) override {
        Viewport viewport(0, 0, window->framebufferSize().x,
                          window->framebufferSize().y);
        CameraState camera{.origin = Vector3F(0, 3, 1),
                           .lookAt = Vector3F(0, 0, -1),
                           .viewport = viewport};
        window->setViewController(std::make_shared<OrthoViewController>(
            std::make_shared<OrthoCamera>(camera, -3, 3, -3, 3)));
    }

    void onResetSim() override {
        _rng.seed(0);

        Plane2Ptr plane = std::make_shared<Plane2>(Vector2D(0, 1), Vector2D());
        RigidBodyCollider2Ptr collider =
            std::make_shared<RigidBodyCollider2>(plane);
        collider->setFrictionCoefficient(0.01);

        _solver = ParticleSystemSolver2();
        _solver.setRestitutionCoefficient(0.5);
        _solver.setCollider(collider);

        ParticleSystemData2Ptr particles = _solver.particleSystemData();
        PointParticleEmitter2Ptr emitter =
            std::make_shared<PointParticleEmitter2>(Vector2D(0, 0),
                                                    Vector2D(0, 1), 10.0, 15.0);
        emitter->setMaxNumberOfNewParticlesPerSecond(100);
        emitter->setMaxNumberOfParticles(1000);
        _solver.setEmitter(emitter);
    }

    void onUpdateRenderables() override {
        auto particles = _solver.particleSystemData();
        auto pos2D = particles->positions();

        size_t oldNumParticles = _positions.length();
        size_t newNumParticles = particles->numberOfParticles();
        _positions.resize(newNumParticles);
        _colors.resize(newNumParticles);

        for (size_t i = 0; i < newNumParticles; ++i) {
            Vector3F pt((float)pos2D[i].x, (float)pos2D[i].y, 0);
            _positions[i] = pt;
        }

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = oldNumParticles; i < newNumParticles; ++i) {
            _colors[i] = ColorUtils::makeJet(dist(_rng));
        }

        _renderable->update(_positions, _colors);
    }
};

class PciSph2Example final : public GfxExample {
 public:
    PciSph2Example() : GfxExample(Frame(0, 0.005)) {}

    std::string name() const override { return "2-D PCISPH Example"; }

    void onSetup(Window* window) override {
        onResetSim();

        auto particles = _solver.sphSystemData();
        auto pos2D = particles->positions();
        _positions.resize(particles->numberOfParticles(), Vector3F());
        _colors.resize(particles->numberOfParticles(), Vector4F());
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist2(-1.0f, 1.0f);
        for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
            _positions[i] = Vector3F((float)pos2D[i].x, (float)pos2D[i].y, 0);
            _colors[i] = ColorUtils::makeJet(dist2(rng));
        }

        _renderable = std::make_shared<PointsRenderable>(
            _positions, _colors, 4.0f * window->displayScalingFactor().x);
        window->renderer()->addRenderable(_renderable);
        window->renderer()->setBackgroundColor({0.1f, 0.1f, 0.1f, 1.0f});
        window->setSwapInterval(1);
    }

    void onAdvanceSim(const jet::Frame& frame) override {
        _solver.update(frame);
    }

 private:
    PciSphSolver2 _solver;
    Array1<Vector3F> _positions;
    Array1<Vector4F> _colors;
    PointsRenderablePtr _renderable;

    void onResetView(Window* window) override {
        Viewport viewport(0, 0, window->framebufferSize().x,
                          window->framebufferSize().y);
        CameraState camera{.origin = Vector3F(0, 1, 1),
                           .lookAt = Vector3F(0, 0, -1),
                           .viewport = viewport};
        window->setViewController(std::make_shared<OrthoViewController>(
            std::make_shared<OrthoCamera>(camera, 0, 1, -1, 1)));
    }

    void onResetSim() override {
        const double targetSpacing = 0.025;

        BoundingBox2D domain(Vector2D(), Vector2D(1, 2));

        // Initialize solvers
        _solver = PciSphSolver2();
        _solver.setViscosityCoefficient(0.01);
        _solver.setPseudoViscosityCoefficient(10);
        _solver.setIsUsingFixedSubTimeSteps(true);
        _solver.setNumberOfFixedSubTimeSteps(1);
        _solver.setMaxNumberOfIterations(20);

        SphSystemData2Ptr particles = _solver.sphSystemData();
        particles->setTargetDensity(1000.0);
        particles->setTargetSpacing(targetSpacing);

        // Initialize source
        ImplicitSurfaceSet2Ptr surfaceSet =
            std::make_shared<ImplicitSurfaceSet2>();
        surfaceSet->addExplicitSurface(std::make_shared<Plane2>(
            Vector2D(0, 1), Vector2D(0, 0.25 * domain.height())));
        surfaceSet->addExplicitSurface(std::make_shared<Sphere2>(
            domain.midPoint(), 0.15 * domain.width()));

        BoundingBox2D sourceBound(domain);
        sourceBound.expand(-targetSpacing);

        auto emitter = std::make_shared<VolumeParticleEmitter2>(
            surfaceSet, sourceBound, targetSpacing, Vector2D());
        _solver.setEmitter(emitter);

        // Initialize boundary
        Box2Ptr box = std::make_shared<Box2>(domain);
        box->isNormalFlipped = true;
        RigidBodyCollider2Ptr collider =
            std::make_shared<RigidBodyCollider2>(box);

        // Setup solver
        _solver.setCollider(collider);

        // Update once to initialize
        _solver.update(currentFrame());
    }

    void onUpdateRenderables() override {
        auto particles = _solver.sphSystemData();
        auto pos2D = particles->positions();

        for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
            Vector3F pt((float)pos2D[i].x, (float)pos2D[i].y, 0);
            _positions[i] = pt;
        }

        _renderable->update(_positions, _colors);
    }
};

void makeGfxDemo(const WindowPtr& window) {
    Logging::mute();
    GfxExampleManager::initialize(window);
    GfxExampleManager::addExample<SimpleExample>();
    GfxExampleManager::addExample<PointsExample>();
    GfxExampleManager::addExample<SimpleParticleAnimationExample>();
    GfxExampleManager::addExample<PciSph2Example>();
}
