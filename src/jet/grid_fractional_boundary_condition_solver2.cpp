// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array_utils.h>
#include <jet/cell_centered_scalar_grid.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit.h>
#include <pch.h>
#include <physics_helpers.h>
#include <algorithm>

using namespace jet;

GridFractionalBoundaryConditionSolver2 ::
    GridFractionalBoundaryConditionSolver2() {}

GridFractionalBoundaryConditionSolver2::
    ~GridFractionalBoundaryConditionSolver2() {}

void GridFractionalBoundaryConditionSolver2::constrainVelocity(
    FaceCenteredGrid2* velocity, unsigned int extrapolationDepth) {
    Vector2UZ size = velocity->resolution();
    if (_colliderSdf == nullptr || _colliderSdf->resolution() != size) {
        updateCollider(collider(), size, velocity->gridSpacing(),
                       velocity->origin());
    }

    auto u = velocity->uView();
    auto v = velocity->vView();
    auto uPos = velocity->uPosition();
    auto vPos = velocity->vPosition();

    Array2<double> uTemp(u.size());
    Array2<double> vTemp(v.size());
    Array2<char> uMarker(u.size(), 1);
    Array2<char> vMarker(v.size(), 1);

    Vector2D h = velocity->gridSpacing();

    // Assign collider's velocity first and initialize markers
    velocity->parallelForEachUIndex([&](const Vector2UZ& idx) {
        Vector2D pt = uPos(idx);
        double phi0 = _colliderSdf->sample(pt - Vector2D(0.5 * h.x, 0.0));
        double phi1 = _colliderSdf->sample(pt + Vector2D(0.5 * h.x, 0.0));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            uMarker(idx) = 1;
        } else {
            Vector2D colliderVel = collider()->velocityAt(pt);
            u(idx) = colliderVel.x;
            uMarker(idx) = 0;
        }
    });

    velocity->parallelForEachVIndex([&](const Vector2UZ& idx) {
        Vector2D pt = vPos(idx);
        double phi0 = _colliderSdf->sample(pt - Vector2D(0.0, 0.5 * h.y));
        double phi1 = _colliderSdf->sample(pt + Vector2D(0.0, 0.5 * h.y));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            vMarker(idx) = 1;
        } else {
            Vector2D colliderVel = collider()->velocityAt(pt);
            v(idx) = colliderVel.y;
            vMarker(idx) = 0;
        }
    });

    // Free-slip: Extrapolate fluid velocity into the collider
    extrapolateToRegion(velocity->uView(), uMarker, extrapolationDepth, u);
    extrapolateToRegion(velocity->vView(), vMarker, extrapolationDepth, v);

    // No-flux: project the extrapolated velocity to the collider's surface
    // normal
    velocity->parallelForEachUIndex([&](const Vector2UZ& idx) {
        Vector2D pt = uPos(idx);
        if (isInsideSdf(_colliderSdf->sample(pt))) {
            Vector2D colliderVel = collider()->velocityAt(pt);
            Vector2D vel = velocity->sample(pt);
            Vector2D g = _colliderSdf->gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector2D n = g.normalized();
                Vector2D velr = vel - colliderVel;
                Vector2D velt = projectAndApplyFriction(
                    velr, n, collider()->frictionCoefficient());

                Vector2D velp = velt + colliderVel;
                uTemp(idx) = velp.x;
            } else {
                uTemp(idx) = colliderVel.x;
            }
        } else {
            uTemp(idx) = u(idx);
        }
    });

    velocity->parallelForEachVIndex([&](const Vector2UZ& idx) {
        Vector2D pt = vPos(idx);
        if (isInsideSdf(_colliderSdf->sample(pt))) {
            Vector2D colliderVel = collider()->velocityAt(pt);
            Vector2D vel = velocity->sample(pt);
            Vector2D g = _colliderSdf->gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector2D n = g.normalized();
                Vector2D velr = vel - colliderVel;
                Vector2D velt = projectAndApplyFriction(
                    velr, n, collider()->frictionCoefficient());

                Vector2D velp = velt + colliderVel;
                vTemp(idx) = velp.y;
            } else {
                vTemp(idx) = colliderVel.y;
            }
        } else {
            vTemp(idx) = v(idx);
        }
    });

    // Transfer results
    parallelForEachIndex(u.size(), [&](size_t i, size_t j) { u(i, j) = uTemp(i, j); });
    parallelForEachIndex(v.size(), [&](size_t i, size_t j) { v(i, j) = vTemp(i, j); });

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

ScalarField2Ptr GridFractionalBoundaryConditionSolver2::colliderSdf() const {
    return _colliderSdf;
}

VectorField2Ptr GridFractionalBoundaryConditionSolver2::colliderVelocityField()
    const {
    return _colliderVel;
}

void GridFractionalBoundaryConditionSolver2::onColliderUpdated(
    const Vector2UZ& gridSize, const Vector2D& gridSpacing,
    const Vector2D& gridOrigin) {
    if (_colliderSdf == nullptr) {
        _colliderSdf = std::make_shared<CellCenteredScalarGrid2>();
    }
    _colliderSdf->resize(gridSize, gridSpacing, gridOrigin);

    if (collider() != nullptr) {
        Surface2Ptr surface = collider()->surface();
        ImplicitSurface2Ptr implicitSurface =
            std::dynamic_pointer_cast<ImplicitSurface2>(surface);
        if (implicitSurface == nullptr) {
            implicitSurface = std::make_shared<SurfaceToImplicit2>(surface);
        }

        _colliderSdf->fill([&](const Vector2D& pt) {
            return implicitSurface->signedDistance(pt);
        });

        _colliderVel = CustomVectorField2::builder()
                           .withFunction([&](const Vector2D& x) {
                               return collider()->velocityAt(x);
                           })
                           .withDerivativeResolution(gridSpacing.x)
                           .makeShared();
    } else {
        _colliderSdf->fill(kMaxD);

        _colliderVel =
            CustomVectorField2::builder()
                .withFunction([](const Vector2D&) { return Vector2D(); })
                .withDerivativeResolution(gridSpacing.x)
                .makeShared();
    }
}
