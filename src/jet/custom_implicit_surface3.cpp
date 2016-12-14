// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constants.h>
#include <jet/custom_implicit_surface3.h>
#include <jet/level_set_utils.h>

using namespace jet;

const double kDistanceThreshold = 1e-3;
const double kGradientThreshold = 1e-3;

CustomImplicitSurface3::CustomImplicitSurface3(
    const std::function<double(const Vector3D&)>& func,
    const BoundingBox3D& domain,
    double resolution,
    bool isNormalFlipped)
: ImplicitSurface3(isNormalFlipped)
, _func(func)
, _domain(domain)
, _resolution(resolution) {
}

CustomImplicitSurface3::~CustomImplicitSurface3() {
}

Vector3D CustomImplicitSurface3::closestPoint(
    const Vector3D& otherPoint) const {
    Vector3D pt = otherPoint;
    while (std::fabs(_func(pt)) < kDistanceThreshold) {
        Vector3D g = gradient(pt);

        if (g.length() < kGradientThreshold) {
            break;
        }

        pt += g;
    }
    return pt;
}

double CustomImplicitSurface3::closestDistance(
    const Vector3D& otherPoint) const {
    return std::fabs(signedDistance(otherPoint));
}

bool CustomImplicitSurface3::intersects(const Ray3D& ray) const {
    BoundingBoxRayIntersection3D intersection
        = _domain.getClosestIntersection(ray);

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
        Vector3D pt = ray.pointAt(t);
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

BoundingBox3D CustomImplicitSurface3::boundingBox() const {
    return _domain;
}

double CustomImplicitSurface3::signedDistance(
    const Vector3D& otherPoint) const {
    if (_func) {
        return _func(otherPoint);
    } else {
        return kMaxD;
    }
}

CustomImplicitSurface3::Builder CustomImplicitSurface3::builder() {
    return Builder();
}

Vector3D CustomImplicitSurface3::actualClosestNormal(
    const Vector3D& otherPoint) const {
    Vector3D pt = otherPoint;
    Vector3D g;
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

SurfaceRayIntersection3 CustomImplicitSurface3::actualClosestIntersection(
    const Ray3D& ray) const {
    SurfaceRayIntersection3 result;

    BoundingBoxRayIntersection3D intersection
        = _domain.getClosestIntersection(ray);

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
        Vector3D pt = ray.pointAt(t);
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

Vector3D CustomImplicitSurface3::gradient(const Vector3D& x) const {
    double left = _func(x - Vector3D(0.5 * _resolution, 0.0, 0.0));
    double right = _func(x + Vector3D(0.5 * _resolution, 0.0, 0.0));
    double bottom = _func(x - Vector3D(0.0, 0.5 * _resolution, 0.0));
    double top = _func(x + Vector3D(0.0, 0.5 * _resolution, 0.0));
    double back = _func(x - Vector3D(0.0, 0.0, 0.5 * _resolution));
    double front = _func(x + Vector3D(0.0, 0.0, 0.5 * _resolution));

    return Vector3D(
        (right - left) / _resolution,
        (top - bottom) / _resolution,
        (front - back) / _resolution);
}


CustomImplicitSurface3::Builder&
CustomImplicitSurface3::Builder::withIsNormalFlipped(bool isNormalFlipped) {
    _isNormalFlipped = isNormalFlipped;
    return *this;
}

CustomImplicitSurface3::Builder&
CustomImplicitSurface3::Builder::withSignedDistanceFunction(
    const std::function<double(const Vector3D&)>& func) {
    _func = func;
    return *this;
}

CustomImplicitSurface3::Builder&
CustomImplicitSurface3::Builder::withDomain(
    const BoundingBox3D& domain) {
    _domain = domain;
    return *this;
}

CustomImplicitSurface3::Builder&
CustomImplicitSurface3::Builder::withResolution(double resolution) {
    _resolution = resolution;
    return *this;
}

CustomImplicitSurface3 CustomImplicitSurface3::Builder::build() const {
    return CustomImplicitSurface3(
        _func,
        _domain,
        _resolution,
        _isNormalFlipped);
}
