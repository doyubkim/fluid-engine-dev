// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/array_utils.h>
#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit2.h>
#include <algorithm>

using namespace jet;

static const char kFluid = 1;
static const char kCollider = 0;

GridBlockedBoundaryConditionSolver2::GridBlockedBoundaryConditionSolver2() {
}

void GridBlockedBoundaryConditionSolver2::constrainVelocity(
    FaceCenteredGrid2* velocity,
    unsigned int extrapolationDepth) {
    Size2 size = velocity->resolution();
    if (_marker.size() != size) {
        updateCollider(
            collider(),
            size,
            velocity->gridSpacing(),
            velocity->origin());
    }

    auto u = velocity->uAccessor();
    auto v = velocity->vAccessor();
    auto uPos = velocity->uPosition();
    auto vPos = velocity->vPosition();

    Array2<char> uMarker(u.size(), 1);
    Array2<char> vMarker(v.size(), 1);

    // Assign collider's velocity first and initialize markers
    _marker.forEachIndex([&](size_t i, size_t j) {
        if (_marker(i, j) == kCollider) {
            if ((i > 0 && _marker(i - 1, j) == kCollider) || i == 0) {
                Vector2D colliderVel = collider()->velocityAt(uPos(i, j));
                u(i, j) = colliderVel.x;
                uMarker(i, j) = 0;
            }
            if ((i < size.x - 1 && _marker(i + 1, j) == kCollider)
                || i == size.x - 1) {
                Vector2D colliderVel = collider()->velocityAt(uPos(i + 1, j));
                u(i + 1, j) = colliderVel.x;
                uMarker(i + 1, j) = 0;
            }
            if ((j > 0 && _marker(i, j - 1) == kCollider) || j == 0) {
                Vector2D colliderVel = collider()->velocityAt(vPos(i, j));
                v(i, j) = colliderVel.y;
                vMarker(i, j) = 0;
            }
            if ((j < size.y - 1 && _marker(i, j + 1) == kCollider)
                || j == size.y - 1) {
                Vector2D colliderVel = collider()->velocityAt(vPos(i, j + 1));
                v(i, j + 1) = colliderVel.y;
                vMarker(i, j + 1) = 0;
            }
        }
    });

    // Free-slip: Extrapolate fluid velocity into the collider
    extrapolateToRegion(
        velocity->uConstAccessor(), uMarker, extrapolationDepth, u);
    extrapolateToRegion(
        velocity->vConstAccessor(), vMarker, extrapolationDepth, v);

    // No-flux: project the extrapolated velocity to the collider's surface
    // normal
    velocity->forEachUIndex([&](size_t i, size_t j) {
        if (uMarker(i, j) == 0) {
            Vector2D pt = uPos(i, j);
            Vector2D colliderVel = collider()->velocityAt(pt);
            Vector2D vel = velocity->sample(pt);
            Vector2D g = _colliderSdf.gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector2D n = g.normalized();
                Vector2D velr = vel - colliderVel;
                Vector2D velt = velr.projected(n);
                if (velt.lengthSquared() > 0) {
                    double veln = velr.dot(n);
                    double mu = collider()->frictionCoefficient();
                    velt *= std::max(1 - mu * veln / velt.length(), 0.0);
                }

                Vector2D velp = velt + colliderVel;
                u(i, j) = velp.x;
            } else {
                u(i, j) = colliderVel.x;
            }
        }
    });

    velocity->parallelForEachVIndex([&](size_t i, size_t j) {
        if (vMarker(i, j) == 0) {
            Vector2D pt = vPos(i, j);
            Vector2D colliderVel = collider()->velocityAt(pt);
            Vector2D vel = velocity->sample(pt);
            Vector2D g = _colliderSdf.gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector2D n = g.normalized();
                Vector2D velr = vel - colliderVel;
                Vector2D velt = velr.projected(n);
                if (velt.lengthSquared() > 0) {
                    double veln = velr.dot(n);
                    double mu = collider()->frictionCoefficient();
                    velt *= std::max(1 - mu * veln / velt.length(), 0.0);
                }

                Vector2D velp = velt + colliderVel;
                v(i, j) = velp.y;
            } else {
                v(i, j) = colliderVel.y;
            }
        }
    });

    // No-flux: project the velocity at the marker interface
    _marker.forEachIndex([&](size_t i, size_t j) {
        if (_marker(i, j) == kCollider) {
            if (i > 0 && _marker(i - 1, j) == kFluid) {
                Vector2D colliderVel = collider()->velocityAt(uPos(i, j));
                u(i, j) = colliderVel.x;
            }
            if (i < size.x - 1 && _marker(i + 1, j) == kFluid) {
                Vector2D colliderVel = collider()->velocityAt(uPos(i + 1, j));
                u(i + 1, j) = colliderVel.x;
            }
            if (j > 0 && _marker(i, j - 1) == kFluid) {
                Vector2D colliderVel = collider()->velocityAt(vPos(i, j));
                v(i, j) = colliderVel.y;
            }
            if (j < size.y - 1 && _marker(i, j + 1) == kFluid) {
                Vector2D colliderVel = collider()->velocityAt(vPos(i, j + 1));
                v(i, j + 1) = colliderVel.y;
            }
        }
    });

    // No-flux: Project velocity on the domain boundary if closed
    if (closedDomainBoundaryFlag() & kDirectionLeft) {
        for (size_t j = 0; j < u.size().y; ++j) {
            u(0, j) = 0;
        }
    }
    if (closedDomainBoundaryFlag() & kDirectionRight) {
        for (size_t j = 0; j < u.size().y; ++j) {
            u(u.size().x - 1, j) = 0;
        }
    }
    if (closedDomainBoundaryFlag() & kDirectionDown) {
        for (size_t i = 0; i < v.size().x; ++i) {
            v(i, 0) = 0;
        }
    }
    if (closedDomainBoundaryFlag() & kDirectionUp) {
        for (size_t i = 0; i < v.size().x; ++i) {
            v(i, v.size().y - 1) = 0;
        }
    }
}

const Array2<char>& GridBlockedBoundaryConditionSolver2::marker() const {
    return _marker;
}

void GridBlockedBoundaryConditionSolver2::onColliderUpdated(
    const Size2& gridSize,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin) {
    _marker.resize(gridSize);
    _colliderSdf.resize(gridSize, gridSpacing, gridOrigin);

    if (collider() != nullptr) {
        auto pos = _colliderSdf.dataPosition();

        Surface2Ptr surface = collider()->surface();
        ImplicitSurface2Ptr implicitSurface
            = std::dynamic_pointer_cast<ImplicitSurface2>(surface);
        if (implicitSurface == nullptr) {
            implicitSurface = std::make_shared<SurfaceToImplicit2>(surface);
        }

        _marker.parallelForEachIndex([&](size_t i, size_t j) {
            if (isInsideSdf(implicitSurface->signedDistance(pos(i, j)))) {
                _marker(i, j) = kCollider;
            } else {
                _marker(i, j) = kFluid;
            }
        });

        _colliderSdf.fill([&](const Vector2D& pt) {
            return implicitSurface->signedDistance(pt);
        });
    } else {
        _marker.set(kFluid);
        _colliderSdf.fill(kMaxD);
    }
}
