// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <physics_helpers.h>
#include <jet/array_utils.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit2.h>
#include <algorithm>

using namespace jet;

GridFractionalBoundaryConditionSolver2
::GridFractionalBoundaryConditionSolver2() {
}

GridFractionalBoundaryConditionSolver2::
~GridFractionalBoundaryConditionSolver2() {
}

void GridFractionalBoundaryConditionSolver2::constrainVelocity(
    FaceCenteredGrid2* velocity,
    unsigned int extrapolationDepth) {
    Size2 size = velocity->resolution();
    if (_colliderSdf.resolution() != size) {
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

    Array2<double> uTemp(u.size());
    Array2<double> vTemp(v.size());
    Array2<char> uMarker(u.size(), 1);
    Array2<char> vMarker(v.size(), 1);

    Vector2D h = velocity->gridSpacing();

    // Assign collider's velocity first and initialize markers
    velocity->parallelForEachUIndex([&](size_t i, size_t j) {
        Vector2D pt = uPos(i, j);
        double phi0 = _colliderSdf.sample(pt - Vector2D(0.5 * h.x, 0.0));
        double phi1 = _colliderSdf.sample(pt + Vector2D(0.5 * h.x, 0.0));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            uMarker(i, j) = 1;
        } else {
            Vector2D colliderVel = collider()->velocityAt(pt);
            u(i, j) = colliderVel.x;
            uMarker(i, j) = 0;
        }
    });

    velocity->parallelForEachVIndex([&](size_t i, size_t j) {
        Vector2D pt = vPos(i, j);
        double phi0 = _colliderSdf.sample(pt - Vector2D(0.0, 0.5 * h.y));
        double phi1 = _colliderSdf.sample(pt + Vector2D(0.0, 0.5 * h.y));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            vMarker(i, j) = 1;
        } else {
            Vector2D colliderVel = collider()->velocityAt(pt);
            v(i, j) = colliderVel.y;
            vMarker(i, j) = 0;
        }
    });

    // Free-slip: Extrapolate fluid velocity into the collider
    extrapolateToRegion(
        velocity->uConstAccessor(), uMarker, extrapolationDepth, u);
    extrapolateToRegion(
        velocity->vConstAccessor(), vMarker, extrapolationDepth, v);

    // No-flux: project the extrapolated velocity to the collider's surface
    // normal
    velocity->parallelForEachUIndex([&](size_t i, size_t j) {
        Vector2D pt = uPos(i, j);
        if (isInsideSdf(_colliderSdf.sample(pt))) {
            Vector2D colliderVel = collider()->velocityAt(pt);
            Vector2D vel = velocity->sample(pt);
            Vector2D g = _colliderSdf.gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector2D n = g.normalized();
                Vector2D velr = vel - colliderVel;
                Vector2D velt = projectAndApplyFriction(
                    velr, n, collider()->frictionCoefficient());

                Vector2D velp = velt + colliderVel;
                uTemp(i, j) = velp.x;
            } else {
                uTemp(i, j) = colliderVel.x;
            }
        } else {
            uTemp(i, j) = u(i, j);
        }
    });

    velocity->parallelForEachVIndex([&](size_t i, size_t j) {
        Vector2D pt = vPos(i, j);
        if (isInsideSdf(_colliderSdf.sample(pt))) {
            Vector2D colliderVel = collider()->velocityAt(pt);
            Vector2D vel = velocity->sample(pt);
            Vector2D g = _colliderSdf.gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector2D n = g.normalized();
                Vector2D velr = vel - colliderVel;
                Vector2D velt = projectAndApplyFriction(
                    velr, n, collider()->frictionCoefficient());

                Vector2D velp = velt + colliderVel;
                vTemp(i, j) = velp.y;
            } else {
                vTemp(i, j) = colliderVel.y;
            }
        } else {
            vTemp(i, j) = v(i, j);
        }
    });

    // Transfer results
    u.parallelForEachIndex([&](size_t i, size_t j) {
        u(i, j) = uTemp(i, j);
    });
    v.parallelForEachIndex([&](size_t i, size_t j) {
        v(i, j) = vTemp(i, j);
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

const CellCenteredScalarGrid2&
GridFractionalBoundaryConditionSolver2::colliderSdf() const {
    return _colliderSdf;
}

void GridFractionalBoundaryConditionSolver2::onColliderUpdated(
    const Size2& gridSize,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin) {
    _colliderSdf.resize(gridSize, gridSpacing, gridOrigin);

    if (collider() != nullptr) {
        Surface2Ptr surface = collider()->surface();
        ImplicitSurface2Ptr implicitSurface
            = std::dynamic_pointer_cast<ImplicitSurface2>(surface);
        if (implicitSurface == nullptr) {
            implicitSurface = std::make_shared<SurfaceToImplicit2>(surface);
        }

        _colliderSdf.fill([&](const Vector2D& pt) {
            return implicitSurface->signedDistance(pt);
        });
    } else {
        _colliderSdf.fill(kMaxD);
    }
}
