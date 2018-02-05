// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <physics_helpers.h>
#include <jet/array_utils.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/grid_fractional_boundary_condition_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit3.h>
#include <algorithm>

using namespace jet;

GridFractionalBoundaryConditionSolver3
::GridFractionalBoundaryConditionSolver3() {
}

GridFractionalBoundaryConditionSolver3::
~GridFractionalBoundaryConditionSolver3() {
}

void GridFractionalBoundaryConditionSolver3::constrainVelocity(
    FaceCenteredGrid3* velocity,
    unsigned int extrapolationDepth) {
    Size3 size = velocity->resolution();
    if (_colliderSdf == nullptr || _colliderSdf->resolution() != size) {
        updateCollider(
            collider(),
            size,
            velocity->gridSpacing(),
            velocity->origin());
    }

    auto u = velocity->uAccessor();
    auto v = velocity->vAccessor();
    auto w = velocity->wAccessor();
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
    velocity->parallelForEachUIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = uPos(i, j, k);
        double phi0 = _colliderSdf->sample(pt - Vector3D(0.5 * h.x, 0.0, 0.0));
        double phi1 = _colliderSdf->sample(pt + Vector3D(0.5 * h.x, 0.0, 0.0));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            uMarker(i, j, k) = 1;
        } else {
            Vector3D colliderVel = collider()->velocityAt(pt);
            u(i, j, k) = colliderVel.x;
            uMarker(i, j, k) = 0;
        }
    });

    velocity->parallelForEachVIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = vPos(i, j, k);
        double phi0 = _colliderSdf->sample(pt - Vector3D(0.0, 0.5 * h.y, 0.0));
        double phi1 = _colliderSdf->sample(pt + Vector3D(0.0, 0.5 * h.y, 0.0));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            vMarker(i, j, k) = 1;
        } else {
            Vector3D colliderVel = collider()->velocityAt(pt);
            v(i, j, k) = colliderVel.y;
            vMarker(i, j, k) = 0;
        }
    });

    velocity->parallelForEachWIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = wPos(i, j, k);
        double phi0 = _colliderSdf->sample(pt - Vector3D(0.0, 0.0, 0.5 * h.z));
        double phi1 = _colliderSdf->sample(pt + Vector3D(0.0, 0.0, 0.5 * h.z));
        double frac = fractionInsideSdf(phi0, phi1);
        frac = 1.0 - clamp(frac, 0.0, 1.0);

        if (frac > 0.0) {
            wMarker(i, j, k) = 1;
        } else {
            Vector3D colliderVel = collider()->velocityAt(pt);
            w(i, j, k) = colliderVel.z;
            wMarker(i, j, k) = 0;
        }
    });

    // Free-slip: Extrapolate fluid velocity into the collider
    extrapolateToRegion(
        velocity->uConstAccessor(), uMarker, extrapolationDepth, u);
    extrapolateToRegion(
        velocity->vConstAccessor(), vMarker, extrapolationDepth, v);
    extrapolateToRegion(
        velocity->wConstAccessor(), wMarker, extrapolationDepth, w);

    // No-flux: project the extrapolated velocity to the collider's surface
    // normal
    velocity->parallelForEachUIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = uPos(i, j, k);
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
                uTemp(i, j, k) = velp.x;
            } else {
                uTemp(i, j, k) = colliderVel.x;
            }
        } else {
            uTemp(i, j, k) = u(i, j, k);
        }
    });

    velocity->parallelForEachVIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = vPos(i, j, k);
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
                vTemp(i, j, k) = velp.y;
            } else {
                vTemp(i, j, k) = colliderVel.y;
            }
        } else {
            vTemp(i, j, k) = v(i, j, k);
        }
    });

    velocity->parallelForEachWIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = wPos(i, j, k);
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
                wTemp(i, j, k) = velp.z;
            } else {
                wTemp(i, j, k) = colliderVel.z;
            }
        } else {
            wTemp(i, j, k) = w(i, j, k);
        }
    });

    // Transfer results
    u.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        u(i, j, k) = uTemp(i, j, k);
    });
    v.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        v(i, j, k) = vTemp(i, j, k);
    });
    w.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
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

VectorField3Ptr
GridFractionalBoundaryConditionSolver3::colliderVelocityField() const {
    return _colliderVel;
}

void GridFractionalBoundaryConditionSolver3::onColliderUpdated(
    const Size3& gridSize,
    const Vector3D& gridSpacing,
    const Vector3D& gridOrigin) {
    if (_colliderSdf == nullptr) {
        _colliderSdf = std::make_shared<CellCenteredScalarGrid3>();
    }
    _colliderSdf->resize(gridSize, gridSpacing, gridOrigin);

    if (collider() != nullptr) {
        Surface3Ptr surface = collider()->surface();
        ImplicitSurface3Ptr implicitSurface
            = std::dynamic_pointer_cast<ImplicitSurface3>(surface);
        if (implicitSurface == nullptr) {
            implicitSurface = std::make_shared<SurfaceToImplicit3>(surface);
        }

        _colliderSdf->fill([&](const Vector3D& pt) {
            return implicitSurface->signedDistance(pt);
        });

        _colliderVel = CustomVectorField3::builder()
        .withFunction([&] (const Vector3D& x) {
            return collider()->velocityAt(x);
        })
        .withDerivativeResolution(gridSpacing.x)
        .makeShared();
    } else {
        _colliderSdf->fill(kMaxD);

        _colliderVel = CustomVectorField3::builder()
            .withFunction([] (const Vector3D&) {
                return Vector3D();
            })
            .withDerivativeResolution(gridSpacing.x)
            .makeShared();
    }
}
