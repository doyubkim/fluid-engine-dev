// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constants.h>
#include <jet/custom_implicit_surface2.h>
#include <jet/level_set_utils.h>

using namespace jet;

const double kDistanceThreshold = 1e-2;
const double kGradientThreshold = 1e-2;

CustomImplicitSurface2::CustomImplicitSurface2(
    const std::function<double(const Vector2D&)>& func,
    const BoundingBox2D& domain,
    double resolution,
    bool isNormalFlipped)
: ImplicitSurface2(isNormalFlipped)
, _func(func)
, _domain(domain)
, _resolution(resolution) {
}

CustomImplicitSurface2::~CustomImplicitSurface2() {
}

Vector2D CustomImplicitSurface2::closestPoint(
    const Vector2D& otherPoint) const {
    Vector2D pt = otherPoint;
    while (std::fabs(_func(pt)) < kDistanceThreshold) {
        Vector2D g = gradient(pt);

        if (g.length() < kGradientThreshold) {
            break;
        }

        pt += g;
    }
    return pt;
}

double CustomImplicitSurface2::closestDistance(
    const Vector2D& otherPoint) const {
    return std::fabs(signedDistance(otherPoint));
}

bool CustomImplicitSurface2::intersects(const Ray2D& ray) const {
    BoundingBoxRayIntersection2D intersection;
    _domain.getClosestIntersection(ray, &intersection);

    if (intersection.isIntersecting) {
        double tStart, tEnd;
        if (intersection.tFar == kMaxD) {
            tStart = 0.0;
            tEnd = intersection.tNear;
        } else {
            tStart = intersection.tNear;
            tEnd = intersection.tFar;
        }

        double t = tStart;
        Vector2D pt = ray.pointAt(t);
        double prevSign = sign(_func(pt));

        while (t <= tEnd) {
            pt = ray.pointAt(t);
            double newSign = sign(_func(pt));

            if (newSign * prevSign < 0.0) {
                return true;
            }

            t += _resolution;
        }
    }

    return false;
}

BoundingBox2D CustomImplicitSurface2::boundingBox() const {
    return _domain;
}

double CustomImplicitSurface2::signedDistance(
    const Vector2D& otherPoint) const {
    if (_func) {
        return _func(otherPoint);
    } else {
        return kMaxD;
    }
}

CustomImplicitSurface2::Builder CustomImplicitSurface2::builder() {
    return Builder();
}

Vector2D CustomImplicitSurface2::actualClosestNormal(
    const Vector2D& otherPoint) const {
    Vector2D pt = otherPoint;
    Vector2D g;
    while (std::fabs(_func(pt)) < kDistanceThreshold) {
        g = gradient(pt);

        if (g.length() < kGradientThreshold) {
            break;
        }

        pt += g;
    }

    if (g.length() > 0.0) {
        g.normalize();
    }

    return g;
}

SurfaceRayIntersection2 CustomImplicitSurface2::actualClosestIntersection(
    const Ray2D& ray) const {
    SurfaceRayIntersection2 result;

    BoundingBoxRayIntersection2D intersection;
    _domain.getClosestIntersection(ray, &intersection);

    if (intersection.isIntersecting) {
        double tStart, tEnd;
        if (intersection.tFar == kMaxD) {
            tStart = 0.0;
            tEnd = intersection.tNear;
        } else {
            tStart = intersection.tNear;
            tEnd = intersection.tFar;
        }

        double t = tStart;
        Vector2D pt = ray.pointAt(t);
        double prevPhi = _func(pt);

        while (t <= tEnd) {
            pt = ray.pointAt(t);
            double newPhi = _func(pt);

            if (newPhi * prevPhi < 0.0) {
                double frac = fractionInsideSdf(prevPhi, newPhi);
                double tSub = t + _resolution * frac;

                result.isIntersecting = true;
                result.t = tSub;
                result.point = ray.pointAt(tSub);
                result.normal = gradient(result.point);
                if (result.normal.length() > 0.0) {
                    result.normal.normalize();
                }

                return result;
            }

            t += _resolution;
        }
    }

    return result;
}

Vector2D CustomImplicitSurface2::gradient(const Vector2D& x) const {
    double left = _func(x - Vector2D(0.5 * _resolution, 0.0));
    double right = _func(x + Vector2D(0.5 * _resolution, 0.0));
    double bottom = _func(x - Vector2D(0.0, 0.5 * _resolution));
    double top = _func(x + Vector2D(0.0, 0.5 * _resolution));

    return Vector2D(
        (right - left) / _resolution,
        (top - bottom) / _resolution);
}


CustomImplicitSurface2::Builder&
CustomImplicitSurface2::Builder::withIsNormalFlipped(bool isNormalFlipped) {
    _isNormalFlipped = isNormalFlipped;
    return *this;
}

CustomImplicitSurface2::Builder&
CustomImplicitSurface2::Builder::withSignedDistanceFunction(
    const std::function<double(const Vector2D&)>& func) {
    _func = func;
    return *this;
}

CustomImplicitSurface2::Builder&
CustomImplicitSurface2::Builder::withDomain(
    const BoundingBox2D& domain) {
    _domain = domain;
    return *this;
}

CustomImplicitSurface2::Builder&
CustomImplicitSurface2::Builder::withResolution(double resolution) {
    _resolution = resolution;
    return *this;
}

CustomImplicitSurface2 CustomImplicitSurface2::Builder::build() const {
    return CustomImplicitSurface2(
        _func,
        _domain,
        _resolution,
        _isNormalFlipped);
}
