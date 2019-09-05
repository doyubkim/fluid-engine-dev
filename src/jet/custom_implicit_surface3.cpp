// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constants.h>
#include <jet/custom_implicit_surface3.h>
#include <jet/level_set_utils.h>

using namespace jet;

CustomImplicitSurface3::CustomImplicitSurface3(
    const std::function<double(const Vector3D&)>& func,
    const BoundingBox3D& domain, double resolution,
    double rayMarchingResolution, unsigned int maxNumOfIterations,
    const Transform3& transform, bool isNormalFlipped)
    : ImplicitSurface3(transform, isNormalFlipped),
      _func(func),
      _domain(domain),
      _resolution(resolution),
      _rayMarchingResolution(rayMarchingResolution),
      _maxNumOfIterations(maxNumOfIterations) {}

CustomImplicitSurface3::~CustomImplicitSurface3() {}

Vector3D CustomImplicitSurface3::closestPointLocal(
    const Vector3D& otherPoint) const {
    Vector3D pt = clamp(otherPoint, _domain.lowerCorner, _domain.upperCorner);
    for (unsigned int iter = 0; iter < _maxNumOfIterations; ++iter) {
        double sdf = signedDistanceLocal(pt);
        if (std::fabs(sdf) < kEpsilonD) {
            break;
        }
        Vector3D g = gradientLocal(pt);
        pt = pt - sdf * g;
    }
    return pt;
}

bool CustomImplicitSurface3::intersectsLocal(const Ray3D& ray) const {
    BoundingBoxRayIntersection3D intersection =
        _domain.closestIntersection(ray);

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
            const double newPhi = _func(pt);
            const double newPhiAbs = std::fabs(newPhi);

            if (newPhi * prevPhi < 0.0) {
                return true;
            }

            t += std::max(newPhiAbs, _rayMarchingResolution);
            prevPhi = newPhi;
        }
    }

    return false;
}

BoundingBox3D CustomImplicitSurface3::boundingBoxLocal() const {
    return _domain;
}

double CustomImplicitSurface3::signedDistanceLocal(
    const Vector3D& otherPoint) const {
    if (_func) {
        return _func(otherPoint);
    } else {
        return kMaxD;
    }
}

Vector3D CustomImplicitSurface3::closestNormalLocal(
    const Vector3D& otherPoint) const {
    Vector3D pt = closestPointLocal(otherPoint);
    Vector3D g = gradientLocal(pt);
    if (g.lengthSquared() > 0.0) {
        return g.normalized();
    } else {
        return g;
    }
}

SurfaceRayIntersection3 CustomImplicitSurface3::closestIntersectionLocal(
    const Ray3D& ray) const {
    SurfaceRayIntersection3 result;

    BoundingBoxRayIntersection3D intersection =
        _domain.closestIntersection(ray);

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
        double tPrev = t;
        Vector3D pt = ray.pointAt(t);
        double prevPhi = _func(pt);

        while (t <= tEnd) {
            pt = ray.pointAt(t);
            const double newPhi = _func(pt);
            const double newPhiAbs = std::fabs(newPhi);

            if (newPhi * prevPhi < 0.0) {
                const double frac = prevPhi / (prevPhi - newPhi);
                const double tSub = tPrev + _rayMarchingResolution * frac;

                result.isIntersecting = true;
                result.distance = tSub;
                result.point = ray.pointAt(tSub);
                result.normal = gradientLocal(result.point);
                if (result.normal.length() > 0.0) {
                    result.normal.normalize();
                }

                return result;
            }

            tPrev = t;
            t += std::max(newPhiAbs, _rayMarchingResolution);
            prevPhi = newPhi;
        }
    }

    return result;
}

Vector3D CustomImplicitSurface3::gradientLocal(const Vector3D& x) const {
    double left = _func(x - Vector3D(0.5 * _resolution, 0.0, 0.0));
    double right = _func(x + Vector3D(0.5 * _resolution, 0.0, 0.0));
    double bottom = _func(x - Vector3D(0.0, 0.5 * _resolution, 0.0));
    double top = _func(x + Vector3D(0.0, 0.5 * _resolution, 0.0));
    double back = _func(x - Vector3D(0.0, 0.0, 0.5 * _resolution));
    double front = _func(x + Vector3D(0.0, 0.0, 0.5 * _resolution));

    return Vector3D((right - left) / _resolution, (top - bottom) / _resolution,
                    (front - back) / _resolution);
}

CustomImplicitSurface3::Builder CustomImplicitSurface3::builder() {
    return Builder();
}

CustomImplicitSurface3::Builder&
CustomImplicitSurface3::Builder::withSignedDistanceFunction(
    const std::function<double(const Vector3D&)>& func) {
    _func = func;
    return *this;
}

CustomImplicitSurface3::Builder& CustomImplicitSurface3::Builder::withDomain(
    const BoundingBox3D& domain) {
    _domain = domain;
    return *this;
}

CustomImplicitSurface3::Builder&
CustomImplicitSurface3::Builder::withResolution(double resolution) {
    _resolution = resolution;
    return *this;
}

CustomImplicitSurface3::Builder&
CustomImplicitSurface3::Builder::withRayMarchingResolution(double resolution) {
    _rayMarchingResolution = resolution;
    return *this;
}

CustomImplicitSurface3::Builder&
CustomImplicitSurface3::Builder::withMaxNumberOfIterations(
    unsigned int numIter) {
    _maxNumOfIterations = numIter;
    return *this;
}

CustomImplicitSurface3 CustomImplicitSurface3::Builder::build() const {
    return CustomImplicitSurface3(_func, _domain, _resolution,
                                  _rayMarchingResolution, _maxNumOfIterations,
                                  _transform, _isNormalFlipped);
}

CustomImplicitSurface3Ptr CustomImplicitSurface3::Builder::makeShared() const {
    return std::shared_ptr<CustomImplicitSurface3>(
        new CustomImplicitSurface3(_func, _domain, _resolution,
                                   _rayMarchingResolution, _maxNumOfIterations,
                                   _transform, _isNormalFlipped),
        [](CustomImplicitSurface3* obj) { delete obj; });
}
