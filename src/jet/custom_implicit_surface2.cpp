// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constants.h>
#include <jet/custom_implicit_surface2.h>
#include <jet/level_set_utils.h>

using namespace jet;

CustomImplicitSurface2::CustomImplicitSurface2(
    const std::function<double(const Vector2D&)>& func,
    const BoundingBox2D& domain,
    double resolution,
    unsigned int maxNumOfIterations,
    const Transform2& transform,
    bool isNormalFlipped)
: ImplicitSurface2(transform, isNormalFlipped)
, _func(func)
, _domain(domain)
, _resolution(resolution)
, _maxNumOfIterations(maxNumOfIterations) {
}

CustomImplicitSurface2::~CustomImplicitSurface2() {
}

Vector2D CustomImplicitSurface2::closestPointLocal(
    const Vector2D& otherPoint) const {
    Vector2D pt = otherPoint;
    for (unsigned int iter = 0; iter < _maxNumOfIterations; ++iter) {
        double sdf = signedDistanceLocal(pt);
        if (std::fabs(sdf) < kEpsilonD) {
            break;
        }
        Vector2D g = gradientLocal(pt);
        pt = pt - sdf * g;
    }
    return pt;
}

bool CustomImplicitSurface2::intersectsLocal(const Ray2D& ray) const {
    BoundingBoxRayIntersection2D intersection
        = _domain.closestIntersection(ray);

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

BoundingBox2D CustomImplicitSurface2::boundingBoxLocal() const {
    return _domain;
}

double CustomImplicitSurface2::signedDistanceLocal(
    const Vector2D& otherPoint) const {
    if (_func) {
        return _func(otherPoint);
    } else {
        return kMaxD;
    }
}

Vector2D CustomImplicitSurface2::closestNormalLocal(
    const Vector2D& otherPoint) const {
    Vector2D pt = closestPointLocal(otherPoint);
    Vector2D g = gradientLocal(pt);
    if (g.lengthSquared() > 0.0) {
        return g.normalized();
    } else {
        return g;
    }
}

SurfaceRayIntersection2 CustomImplicitSurface2::closestIntersectionLocal(
    const Ray2D& ray) const {
    SurfaceRayIntersection2 result;

    BoundingBoxRayIntersection2D intersection
        = _domain.closestIntersection(ray);

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
                result.normal = gradientLocal(result.point);
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

Vector2D CustomImplicitSurface2::gradientLocal(const Vector2D& x) const {
    double left = _func(x - Vector2D(0.5 * _resolution, 0.0));
    double right = _func(x + Vector2D(0.5 * _resolution, 0.0));
    double bottom = _func(x - Vector2D(0.0, 0.5 * _resolution));
    double top = _func(x + Vector2D(0.0, 0.5 * _resolution));

    return Vector2D(
        (right - left) / _resolution,
        (top - bottom) / _resolution);
}

CustomImplicitSurface2::Builder CustomImplicitSurface2::builder() {
    return Builder();
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

CustomImplicitSurface2::Builder&
CustomImplicitSurface2::Builder::withMaxNumberOfIterations(
    unsigned int numIter) {
    _maxNumOfIterations = numIter;
    return *this;
}

CustomImplicitSurface2 CustomImplicitSurface2::Builder::build() const {
    return CustomImplicitSurface2(
        _func,
        _domain,
        _resolution,
        _maxNumOfIterations,
        _transform,
        _isNormalFlipped);
}

CustomImplicitSurface2Ptr CustomImplicitSurface2::Builder::makeShared() const {
    return std::shared_ptr<CustomImplicitSurface2>(
        new CustomImplicitSurface2(
            _func,
            _domain,
            _resolution,
        _maxNumOfIterations,
            _transform,
            _isNormalFlipped),
        [] (CustomImplicitSurface2* obj) {
            delete obj;
        });
}
