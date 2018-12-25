// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array_utils.h>
#include <jet/cell_centered_scalar_grid.h>
#include <jet/grid_fractional_boundary_condition_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit.h>
#include <common.h>
#include <physics_helpers.h>
#include <algorithm>

using namespace jet;

GridFractionalBoundaryConditionSolver3 ::
    GridFractionalBoundaryConditionSolver3() {}

GridFractionalBoundaryConditionSolver3::
    ~GridFractionalBoundaryConditionSolver3() {}

void GridFractionalBoundaryConditionSolver3::constrainVelocity(
    FaceCenteredGrid3* velocity, unsigned int extrapolationDepth) {
    Vector3UZ size = velocity->resolution();
    if (_colliderSdf == nullptr || _colliderSdf->resolution() != size) {
        updateCollider(collider(), size, velocity->gridSpacing(),
                       velocity->origin());
    }

    auto u = velocity->uView();
    auto v = velocity->vView();
    auto w = velocity->wView();
    auto uPos = velocity->uPosition();
    auto vPos = velocity->vPosition();
    auto wPos = velocity->wPosition();

    Array3<double> uTemp(u.size());
    Array3<double> vTemp(v.size());
    Array3<double> wTemp(w.size());
    Array3<char> uMarker(u.size(), 1);
    Array3<char> vMarker(v.size(), 1);
    Array3<char> wMarker(w.size(), 1);

    Vector3D h = velocity->gridSpacing();

    // Assign collider's velocity first and initialize markers
    velocity->parallelForEachUIndex([&](const Vector3UZ& idx) {
        Vector3D pt = uPos(idx);
        double phi0 = _colliderSdf->sample(pt - Vector3D(0.5 * h.x, 0.0, 0.0));
        double phi1 = _colliderSdf->sample(pt + Vector3D(0.5 * h.x, 0.0, 0.0));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            uMarker(idx) = 1;
        } else {
            Vector3D colliderVel = collider()->velocityAt(pt);
            u(idx) = colliderVel.x;
            uMarker(idx) = 0;
        }
    });

    velocity->parallelForEachVIndex([&](const Vector3UZ& idx) {
        Vector3D pt = vPos(idx);
        double phi0 = _colliderSdf->sample(pt - Vector3D(0.0, 0.5 * h.y, 0.0));
        double phi1 = _colliderSdf->sample(pt + Vector3D(0.0, 0.5 * h.y, 0.0));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            vMarker(idx) = 1;
        } else {
            Vector3D colliderVel = collider()->velocityAt(pt);
            v(idx) = colliderVel.y;
            vMarker(idx) = 0;
        }
    });

    velocity->parallelForEachWIndex([&](const Vector3UZ& idx) {
        Vector3D pt = wPos(idx);
        double phi0 = _colliderSdf->sample(pt - Vector3D(0.0, 0.0, 0.5 * h.z));
        double phi1 = _colliderSdf->sample(pt + Vector3D(0.0, 0.0, 0.5 * h.z));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            wMarker(idx) = 1;
        } else {
            Vector3D colliderVel = collider()->velocityAt(pt);
            w(idx) = colliderVel.z;
            wMarker(idx) = 0;
        }
    });

    // Free-slip: Extrapolate fluid velocity into the collider
    extrapolateToRegion(velocity->uView(), uMarker, extrapolationDepth, u);
    extrapolateToRegion(velocity->vView(), vMarker, extrapolationDepth, v);
    extrapolateToRegion(velocity->wView(), wMarker, extrapolationDepth, w);

    // No-flux: project the extrapolated velocity to the collider's surface
    // normal
    velocity->parallelForEachUIndex([&](const Vector3UZ& idx) {
        Vector3D pt = uPos(idx);
        if (isInsideSdf(_colliderSdf->sample(pt))) {
            Vector3D colliderVel = collider()->velocityAt(pt);
            Vector3D vel = velocity->sample(pt);
            Vector3D g = _colliderSdf->gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector3D n = g.normalized();
                Vector3D velr = vel - colliderVel;
                Vector3D velt = projectAndApplyFriction(
                    velr, n, collider()->frictionCoefficient());

                Vector3D velp = velt + colliderVel;
                uTemp(idx) = velp.x;
            } else {
                uTemp(idx) = colliderVel.x;
            }
        } else {
            uTemp(idx) = u(idx);
        }
    });

    velocity->parallelForEachVIndex([&](const Vector3UZ& idx) {
        Vector3D pt = vPos(idx);
        if (isInsideSdf(_colliderSdf->sample(pt))) {
            Vector3D colliderVel = collider()->velocityAt(pt);
            Vector3D vel = velocity->sample(pt);
            Vector3D g = _colliderSdf->gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector3D n = g.normalized();
                Vector3D velr = vel - colliderVel;
                Vector3D velt = projectAndApplyFriction(
                    velr, n, collider()->frictionCoefficient());

                Vector3D velp = velt + colliderVel;
                vTemp(idx) = velp.y;
            } else {
                vTemp(idx) = colliderVel.y;
            }
        } else {
            vTemp(idx) = v(idx);
        }
    });

    velocity->parallelForEachWIndex([&](const Vector3UZ& idx) {
        Vector3D pt = wPos(idx);
        if (isInsideSdf(_colliderSdf->sample(pt))) {
            Vector3D colliderVel = collider()->velocityAt(pt);
            Vector3D vel = velocity->sample(pt);
            Vector3D g = _colliderSdf->gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector3D n = g.normalized();
                Vector3D velr = vel - colliderVel;
                Vector3D velt = projectAndApplyFriction(
                    velr, n, collider()->frictionCoefficient());

                Vector3D velp = velt + colliderVel;
                wTemp(idx) = velp.z;
            } else {
                wTemp(idx) = colliderVel.z;
            }
        } else {
            wTemp(idx) = w(idx);
        }
    });

    // Transfer results
    parallelForEachIndex(u.size(), [&](size_t i, size_t j, size_t k) {
        u(i, j, k) = uTemp(i, j, k);
    });
    parallelForEachIndex(v.size(), [&](size_t i, size_t j, size_t k) {
        v(i, j, k) = vTemp(i, j, k);
    });
    parallelForEachIndex(w.size(), [&](size_t i, size_t j, size_t k) {
        w(i, j, k) = wTemp(i, j, k);
    });

    // No-flux: Project velocity on the domain boundary if closed
    if (closedDomainBoundaryFlag() & kDirectionLeft) {
        for (size_t k = 0; k < u.size().z; ++k) {
            for (size_t j = 0; j < u.size().y; ++j) {
                u(0, j, k) = 0;
            }
        }
    }
    if (closedDomainBoundaryFlag() & kDirectionRight) {
        for (size_t k = 0; k < u.size().z; ++k) {
            for (size_t j = 0; j < u.size().y; ++j) {
                u(u.size().x - 1, j, k) = 0;
            }
        }
    }
    if (closedDomainBoundaryFlag() & kDirectionDown) {
        for (size_t k = 0; k < v.size().z; ++k) {
            for (size_t i = 0; i < v.size().x; ++i) {
                v(i, 0, k) = 0;
            }
        }
    }
    if (closedDomainBoundaryFlag() & kDirectionUp) {
        for (size_t k = 0; k < v.size().z; ++k) {
            for (size_t i = 0; i < v.size().x; ++i) {
                v(i, v.size().y - 1, k) = 0;
            }
        }
    }
    if (closedDomainBoundaryFlag() & kDirectionBack) {
        for (size_t j = 0; j < w.size().y; ++j) {
            for (size_t i = 0; i < w.size().x; ++i) {
                w(i, j, 0) = 0;
            }
        }
    }
    if (closedDomainBoundaryFlag() & kDirectionFront) {
        for (size_t j = 0; j < w.size().y; ++j) {
            for (size_t i = 0; i < w.size().x; ++i) {
                w(i, j, w.size().z - 1) = 0;
            }
        }
    }
}

ScalarField3Ptr GridFractionalBoundaryConditionSolver3::colliderSdf() const {
    return _colliderSdf;
}

VectorField3Ptr GridFractionalBoundaryConditionSolver3::colliderVelocityField()
    const {
    return _colliderVel;
}

void GridFractionalBoundaryConditionSolver3::onColliderUpdated(
    const Vector3UZ& gridSize, const Vector3D& gridSpacing,
    const Vector3D& gridOrigin) {
    if (_colliderSdf == nullptr) {
        _colliderSdf = std::make_shared<CellCenteredScalarGrid3>();
    }
    _colliderSdf->resize(gridSize, gridSpacing, gridOrigin);

    if (collider() != nullptr) {
        Surface3Ptr surface = collider()->surface();
        ImplicitSurface3Ptr implicitSurface =
            std::dynamic_pointer_cast<ImplicitSurface3>(surface);
        if (implicitSurface == nullptr) {
            implicitSurface = std::make_shared<SurfaceToImplicit3>(surface);
        }

        _colliderSdf->fill([&](const Vector3D& pt) {
            return implicitSurface->signedDistance(pt);
        });

        _colliderVel = CustomVectorField3::builder()
                           .withFunction([&](const Vector3D& x) {
                               return collider()->velocityAt(x);
                           })
                           .withDerivativeResolution(gridSpacing.x)
                           .makeShared();
    } else {
        _colliderSdf->fill(kMaxD);

        _colliderVel =
            CustomVectorField3::builder()
                .withFunction([](const Vector3D&) { return Vector3D(); })
                .withDerivativeResolution(gridSpacing.x)
                .makeShared();
    }
}
