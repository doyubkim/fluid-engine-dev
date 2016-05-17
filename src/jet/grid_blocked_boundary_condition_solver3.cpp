// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/array_utils.h>
#include <jet/grid_blocked_boundary_condition_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit3.h>
#include <algorithm>

using namespace jet;

static const char kFluid = 1;
static const char kCollider = 0;

GridBlockedBoundaryConditionSolver3::GridBlockedBoundaryConditionSolver3() {
}

void GridBlockedBoundaryConditionSolver3::constrainVelocity(
    FaceCenteredGrid3* velocity,
    unsigned int extrapolationDepth) {
    Size3 size = velocity->resolution();
    if (_marker.size() != size) {
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

    Array3<char> uMarker(u.size(), 1);
    Array3<char> vMarker(v.size(), 1);
    Array3<char> wMarker(w.size(), 1);

    // Assign collider's velocity first and initialize markers
    _marker.forEachIndex([&](size_t i, size_t j, size_t k) {
        if (_marker(i, j, k) == kCollider) {
            if ((i > 0 && _marker(i - 1, j, k) == kCollider) || i == 0) {
                Vector3D colliderVel = collider()->velocityAt(uPos(i, j, k));
                u(i, j, k) = colliderVel.x;
                uMarker(i, j, k) = 0;
            }
            if ((i < size.x - 1 && _marker(i + 1, j, k) == kCollider)
                || i == size.x - 1) {
                Vector3D colliderVel
                    = collider()->velocityAt(uPos(i + 1, j, k));
                u(i + 1, j, k) = colliderVel.x;
                uMarker(i + 1, j, k) = 0;
            }
            if ((j > 0 && _marker(i, j - 1, k) == kCollider) || j == 0) {
                Vector3D colliderVel = collider()->velocityAt(vPos(i, j, k));
                v(i, j, k) = colliderVel.y;
                vMarker(i, j, k) = 0;
            }
            if ((j < size.y - 1 && _marker(i, j + 1, k) == kCollider)
                || j == size.y - 1) {
                Vector3D colliderVel
                    = collider()->velocityAt(vPos(i, j + 1, k));
                v(i, j + 1, k) = colliderVel.y;
                vMarker(i, j + 1, k) = 0;
            }
            if ((k > 0 && _marker(i, j, k - 1) == kCollider) || j == 0) {
                Vector3D colliderVel = collider()->velocityAt(wPos(i, j, k));
                w(i, j, k) = colliderVel.z;
                vMarker(i, j, k) = 0;
            }
            if ((k < size.z - 1 && _marker(i, j, k + 1) == kCollider)
                || k == size.z - 1) {
                Vector3D colliderVel
                    = collider()->velocityAt(wPos(i, j, k + 1));
                w(i, j, k + 1) = colliderVel.z;
                wMarker(i, j, k + 1) = 0;
            }
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
        if (uMarker(i, j, k) == 0) {
            Vector3D pt = uPos(i, j, k);
            Vector3D colliderVel = collider()->velocityAt(pt);
            Vector3D vel = velocity->sample(pt);
            Vector3D g = _colliderSdf.gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector3D n = g.normalized();
                Vector3D velr = vel - colliderVel;
                Vector3D velt = velr.projected(n);
                if (velt.lengthSquared() > 0) {
                    double veln = velr.dot(n);
                    double mu = collider()->frictionCoefficient();
                    velt *= std::max(1 - mu * veln / velt.length(), 0.0);
                }

                Vector3D velp = velt + colliderVel;
                u(i, j, k) = velp.x;
            } else {
                u(i, j, k) = colliderVel.x;
            }
        }
    });

    velocity->parallelForEachVIndex([&](size_t i, size_t j, size_t k) {
        if (vMarker(i, j, k) == 0) {
            Vector3D pt = vPos(i, j, k);
            Vector3D colliderVel = collider()->velocityAt(pt);
            Vector3D vel = velocity->sample(pt);
            Vector3D g = _colliderSdf.gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector3D n = g.normalized();
                Vector3D velr = vel - colliderVel;
                Vector3D velt = velr.projected(n);
                if (velt.lengthSquared() > 0) {
                    double veln = velr.dot(n);
                    double mu = collider()->frictionCoefficient();
                    velt *= std::max(1 - mu * veln / velt.length(), 0.0);
                }

                Vector3D velp = velt + colliderVel;
                v(i, j, k) = velp.y;
            } else {
                v(i, j, k) = colliderVel.y;
            }
        }
    });

    velocity->parallelForEachWIndex([&](size_t i, size_t j, size_t k) {
        if (wMarker(i, j, k) == 0) {
            Vector3D pt = wPos(i, j, k);
            Vector3D colliderVel = collider()->velocityAt(pt);
            Vector3D vel = velocity->sample(pt);
            Vector3D g = _colliderSdf.gradient(pt);
            if (g.lengthSquared() > 0.0) {
                Vector3D n = g.normalized();
                Vector3D velr = vel - colliderVel;
                Vector3D velt = velr.projected(n);
                if (velt.lengthSquared() > 0) {
                    double veln = velr.dot(n);
                    double mu = collider()->frictionCoefficient();
                    velt *= std::max(1 - mu * veln / velt.length(), 0.0);
                }

                Vector3D velp = velt + colliderVel;
                w(i, j, k) = velp.z;
            } else {
                w(i, j, k) = colliderVel.z;
            }
        }
    });

    // No-flux: project the velocity at the marker interface
    _marker.forEachIndex([&](size_t i, size_t j, size_t k) {
        if (_marker(i, j, k) == kCollider) {
            if (i > 0 && _marker(i - 1, j, k) == kFluid) {
                Vector3D colliderVel = collider()->velocityAt(uPos(i, j, k));
                u(i, j, k) = colliderVel.x;
            }
            if (i < size.x - 1 && _marker(i + 1, j, k) == kFluid) {
                Vector3D colliderVel
                    = collider()->velocityAt(uPos(i + 1, j, k));
                u(i + 1, j, k) = colliderVel.x;
            }
            if (j > 0 && _marker(i, j - 1, k) == kFluid) {
                Vector3D colliderVel = collider()->velocityAt(vPos(i, j, k));
                v(i, j, k) = colliderVel.y;
            }
            if (j < size.y - 1 && _marker(i, j + 1, k) == kFluid) {
                Vector3D colliderVel
                    = collider()->velocityAt(vPos(i, j + 1, k));
                v(i, j + 1, k) = colliderVel.y;
            }
            if (k > 0 && _marker(i, j, k - 1) == kFluid) {
                Vector3D colliderVel = collider()->velocityAt(wPos(i, j, k));
                w(i, j, k) = colliderVel.z;
            }
            if (k < size.z - 1 && _marker(i, j, k + 1) == kFluid) {
                Vector3D colliderVel
                    = collider()->velocityAt(wPos(i, j, k + 1));
                w(i, j, k + 1) = colliderVel.z;
            }
        }
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

const Array3<char>& GridBlockedBoundaryConditionSolver3::marker() const {
    return _marker;
}

void GridBlockedBoundaryConditionSolver3::onColliderUpdated(
    const Size3& gridSize,
    const Vector3D& gridSpacing,
    const Vector3D& gridOrigin) {
    _marker.resize(gridSize);
    _colliderSdf.resize(gridSize, gridSpacing, gridOrigin);

    if (collider() != nullptr) {
        auto pos = _colliderSdf.dataPosition();

        Surface3Ptr surface = collider()->surface();
        ImplicitSurface3Ptr implicitSurface
            = std::dynamic_pointer_cast<ImplicitSurface3>(surface);
        if (implicitSurface == nullptr) {
            implicitSurface = std::make_shared<SurfaceToImplicit3>(surface);
        }

        _marker.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
            if (isInsideSdf(implicitSurface->signedDistance(pos(i, j, k)))) {
                _marker(i, j, k) = kCollider;
            } else {
                _marker(i, j, k) = kFluid;
            }
        });

        _colliderSdf.fill([&](const Vector3D& pt) {
            return implicitSurface->signedDistance(pt);
        });
    } else {
        _marker.set(kFluid);
        _colliderSdf.fill(kMaxD);
    }
}
