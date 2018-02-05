// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// buildSignedDistanceField function is adopted from Christopher Batty's code
// https://cs.uwaterloo.ca/~c2batty/code/variationalplusgfm.zip
//

#include <pch.h>
#include <jet/array_utils.h>
#include <jet/level_set_utils.h>
#include <jet/pic_solver2.h>
#include <jet/timer.h>
#include <algorithm>

using namespace jet;

PicSolver2::PicSolver2() : PicSolver2({1, 1}, {1, 1}, {0, 0}) {
}

PicSolver2::PicSolver2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin)
: GridFluidSolver2(resolution, gridSpacing, gridOrigin) {
    auto grids = gridSystemData();
    _signedDistanceFieldId = grids->addScalarData(
        std::make_shared<CellCenteredScalarGrid2::Builder>(), kMaxD);
    _particles = std::make_shared<ParticleSystemData2>();
}

PicSolver2::~PicSolver2() {
}

ScalarGrid2Ptr PicSolver2::signedDistanceField() const {
    return gridSystemData()->scalarDataAt(_signedDistanceFieldId);
}

const ParticleSystemData2Ptr& PicSolver2::particleSystemData() const {
    return _particles;
}

const ParticleEmitter2Ptr& PicSolver2::particleEmitter() const {
    return _particleEmitter;
}

void PicSolver2::setParticleEmitter(const ParticleEmitter2Ptr& newEmitter) {
    _particleEmitter = newEmitter;
    newEmitter->setTarget(_particles);
}

void PicSolver2::onInitialize() {
    GridFluidSolver2::onInitialize();

    Timer timer;
    updateParticleEmitter(0.0);
    JET_INFO << "Update particle emitter took "
             << timer.durationInSeconds() << " seconds";
}

void PicSolver2::onBeginAdvanceTimeStep(double timeIntervalInSeconds) {
    UNUSED_VARIABLE(timeIntervalInSeconds);

    Timer timer;
    updateParticleEmitter(timeIntervalInSeconds);
    JET_INFO << "Update particle emitter took "
             << timer.durationInSeconds() << " seconds";

    JET_INFO << "Number of PIC-type particles: "
             << _particles->numberOfParticles();

    timer.reset();
    transferFromParticlesToGrids();
    JET_INFO << "transferFromParticlesToGrids took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    buildSignedDistanceField();
    JET_INFO << "buildSignedDistanceField took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    extrapolateVelocityToAir();
    JET_INFO << "extrapolateVelocityToAir took "
             << timer.durationInSeconds() << " seconds";

    applyBoundaryCondition();
}

void PicSolver2::computeAdvection(double timeIntervalInSeconds) {
    Timer timer;
    extrapolateVelocityToAir();
    JET_INFO << "extrapolateVelocityToAir took "
             << timer.durationInSeconds() << " seconds";

    applyBoundaryCondition();

    timer.reset();
    transferFromGridsToParticles();
    JET_INFO << "transferFromGridsToParticles took "
             << timer.durationInSeconds() << " seconds";

    timer.reset();
    moveParticles(timeIntervalInSeconds);
    JET_INFO << "moveParticles took "
             << timer.durationInSeconds() << " seconds";
}

ScalarField2Ptr PicSolver2::fluidSdf() const {
    return signedDistanceField();
}

void PicSolver2::transferFromParticlesToGrids() {
    auto flow = gridSystemData()->velocity();
    auto positions = _particles->positions();
    auto velocities = _particles->velocities();
    size_t numberOfParticles = _particles->numberOfParticles();

    // Clear velocity to zero
    flow->fill(Vector2D());

    // Weighted-average velocity
    auto u = flow->uAccessor();
    auto v = flow->vAccessor();
    Array2<double> uWeight(u.size());
    Array2<double> vWeight(v.size());
    _uMarkers.resize(u.size());
    _vMarkers.resize(v.size());
    _uMarkers.set(0);
    _vMarkers.set(0);
    LinearArraySampler2<double, double> uSampler(
        flow->uConstAccessor(),
        flow->gridSpacing(),
        flow->uOrigin());
    LinearArraySampler2<double, double> vSampler(
        flow->vConstAccessor(),
        flow->gridSpacing(),
        flow->vOrigin());
    for (size_t i = 0; i < numberOfParticles; ++i) {
        std::array<Point2UI, 4> indices;
        std::array<double, 4> weights;

        uSampler.getCoordinatesAndWeights(positions[i], &indices, &weights);
        for (int j = 0; j < 4; ++j) {
            u(indices[j]) += velocities[i].x * weights[j];
            uWeight(indices[j]) += weights[j];
            _uMarkers(indices[j]) = 1;
        }

        vSampler.getCoordinatesAndWeights(positions[i], &indices, &weights);
        for (int j = 0; j < 4; ++j) {
            v(indices[j]) += velocities[i].y * weights[j];
            vWeight(indices[j]) += weights[j];
            _vMarkers(indices[j]) = 1;
        }
    }

    uWeight.parallelForEachIndex([&](size_t i, size_t j) {
        if (uWeight(i, j) > 0.0) {
            u(i, j) /= uWeight(i, j);
        }
    });
    vWeight.parallelForEachIndex([&](size_t i, size_t j) {
        if (vWeight(i, j) > 0.0) {
            v(i, j) /= vWeight(i, j);
        }
    });
}

void PicSolver2::transferFromGridsToParticles() {
    auto flow = gridSystemData()->velocity();
    auto positions = _particles->positions();
    auto velocities = _particles->velocities();
    size_t numberOfParticles = _particles->numberOfParticles();

    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        velocities[i] = flow->sample(positions[i]);
    });
}

void PicSolver2::moveParticles(double timeIntervalInSeconds) {
    auto flow = gridSystemData()->velocity();
    auto positions = _particles->positions();
    auto velocities = _particles->velocities();
    size_t numberOfParticles = _particles->numberOfParticles();
    int domainBoundaryFlag = closedDomainBoundaryFlag();
    BoundingBox2D boundingBox = flow->boundingBox();

    parallelFor(kZeroSize, numberOfParticles, [&](size_t i) {
        Vector2D pt0 = positions[i];
        Vector2D pt1 = pt0;
        Vector2D vel = velocities[i];

        // Adaptive time-stepping
        unsigned int numSubSteps
            = static_cast<unsigned int>(std::max(maxCfl(), 1.0));
        double dt = timeIntervalInSeconds / numSubSteps;
        for (unsigned int t = 0; t < numSubSteps; ++t) {
            Vector2D vel0 = flow->sample(pt0);

            // Mid-point rule
            Vector2D midPt = pt0 + 0.5 * dt * vel0;
            Vector2D midVel = flow->sample(midPt);
            pt1 = pt0 + dt * midVel;

            pt0 = pt1;
        }

        if ((domainBoundaryFlag & kDirectionLeft)
            && pt1.x <= boundingBox.lowerCorner.x) {
            pt1.x = boundingBox.lowerCorner.x;
            vel.x = 0.0;
        }
        if ((domainBoundaryFlag & kDirectionRight)
            && pt1.x >= boundingBox.upperCorner.x) {
            pt1.x = boundingBox.upperCorner.x;
            vel.x = 0.0;
        }
        if ((domainBoundaryFlag & kDirectionDown)
            && pt1.y <= boundingBox.lowerCorner.y) {
            pt1.y = boundingBox.lowerCorner.y;
            vel.y = 0.0;
        }
        if ((domainBoundaryFlag & kDirectionUp)
            && pt1.y >= boundingBox.upperCorner.y) {
            pt1.y = boundingBox.upperCorner.y;
            vel.y = 0.0;
        }

        positions[i] = pt1;
        velocities[i] = vel;
    });

    Collider2Ptr col = collider();
    if (col != nullptr) {
        parallelFor(
            kZeroSize,
            numberOfParticles,
            [&](size_t i) {
                col->resolveCollision(
                    0.0,
                    0.0,
                    &positions[i],
                    &velocities[i]);
            });
    }
}

void PicSolver2::extrapolateVelocityToAir() {
    auto vel = gridSystemData()->velocity();
    auto u = vel->uAccessor();
    auto v = vel->vAccessor();

    unsigned int depth = static_cast<unsigned int>(std::ceil(maxCfl()));
    extrapolateToRegion(vel->uConstAccessor(), _uMarkers, depth, u);
    extrapolateToRegion(vel->vConstAccessor(), _vMarkers, depth, v);
}

void PicSolver2::buildSignedDistanceField() {
    auto sdf = signedDistanceField();
    auto sdfPos = sdf->dataPosition();
    double maxH = std::max(sdf->gridSpacing().x, sdf->gridSpacing().y);
    double radius = 1.2 * maxH / std::sqrt(2.0);

    _particles->buildNeighborSearcher(2 * radius);
    auto searcher = _particles->neighborSearcher();
    sdf->parallelForEachDataPointIndex([&] (size_t i, size_t j) {
        Vector2D pt = sdfPos(i, j);
        double minDist = 2.0 * radius;
        searcher->forEachNearbyPoint(
            pt, 2.0 * radius, [&] (size_t, const Vector2D& x) {
                minDist = std::min(minDist, pt.distanceTo(x));
            });
        (*sdf)(i, j) = minDist - radius;
    });

    extrapolateIntoCollider(sdf.get());
}

void PicSolver2::updateParticleEmitter(double timeIntervalInSeconds) {
    if (_particleEmitter != nullptr) {
        _particleEmitter->update(currentTimeInSeconds(), timeIntervalInSeconds);
    }
}

PicSolver2::Builder PicSolver2::builder() {
    return Builder();
}


PicSolver2 PicSolver2::Builder::build() const {
    return PicSolver2(_resolution, getGridSpacing(), _gridOrigin);
}

PicSolver2Ptr PicSolver2::Builder::makeShared() const {
    return std::shared_ptr<PicSolver2>(
        new PicSolver2(
            _resolution,
            getGridSpacing(),
            _gridOrigin),
        [] (PicSolver2* obj) {
            delete obj;
        });
}
