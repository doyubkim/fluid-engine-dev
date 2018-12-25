// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/constants.h>
#include <jet/custom_implicit_surface.h>
#include <jet/level_set_utils.h>

namespace jet {

template <size_t N>
CustomImplicitSurface<N>::CustomImplicitSurface(
    const std::function<double(const Vector<double, N> &)> &func,
    const BoundingBox<double, N> &domain, double resolution,
    double rayMarchingResolution, unsigned int maxNumOfIterations,
    const Transform<N> &transform, bool isNormalFlipped)
    : ImplicitSurface<N>(transform, isNormalFlipped),
      _func(func),
      _domain(domain),
      _resolution(resolution),
      _rayMarchingResolution(rayMarchingResolution),
      _maxNumOfIterations(maxNumOfIterations) {}

template <size_t N>
CustomImplicitSurface<N>::~CustomImplicitSurface() {}

template <size_t N>
Vector<double, N> CustomImplicitSurface<N>::closestPointLocal(
    const Vector<double, N> &otherPoint) const {
    Vector<double, N> pt =
        clamp(otherPoint, _domain.lowerCorner, _domain.upperCorner);
    for (unsigned int iter = 0; iter < _maxNumOfIterations; ++iter) {
        double sdf = signedDistanceLocal(pt);
        if (std::fabs(sdf) < kEpsilonD) {
            break;
        }
        Vector<double, N> g = gradientLocal(pt);
        pt = pt - sdf * g;
    }
    return pt;
}

template <size_t N>
bool CustomImplicitSurface<N>::intersectsLocal(
    const Ray<double, N> &ray) const {
    BoundingBoxRayIntersectionD intersection = _domain.closestIntersection(ray);

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
        Vector<double, N> pt = ray.pointAt(t);
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

template <size_t N>
BoundingBox<double, N> CustomImplicitSurface<N>::boundingBoxLocal() const {
    return _domain;
}

template <size_t N>
double CustomImplicitSurface<N>::signedDistanceLocal(
    const Vector<double, N> &otherPoint) const {
    if (_func) {
        return _func(otherPoint);
    } else {
        return kMaxD;
    }
}

template <size_t N>
Vector<double, N> CustomImplicitSurface<N>::closestNormalLocal(
    const Vector<double, N> &otherPoint) const {
    Vector<double, N> pt = closestPointLocal(otherPoint);
    Vector<double, N> g = gradientLocal(pt);
    if (g.lengthSquared() > 0.0) {
        return g.normalized();
    } else {
        return g;
    }
}

template <size_t N>
SurfaceRayIntersection<N> CustomImplicitSurface<N>::closestIntersectionLocal(
    const Ray<double, N> &ray) const {
    SurfaceRayIntersection<N> result;

    BoundingBoxRayIntersectionD intersection = _domain.closestIntersection(ray);

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
        Vector<double, N> pt = ray.pointAt(t);
        double prevPhi = _func(pt);

        while (t <= tEnd) {
            pt = ray.pointAt(t);
            const double newPhi = _func(pt);
            const double newPhiAbs = std::fabs(newPhi);

            if (newPhi * prevPhi < 0.0) {
                const double frac = prevPhi / (prevPhi - newPhi);
                const double tSub = t + _rayMarchingResolution * frac;

                result.isIntersecting = true;
                result.distance = tSub;
                result.point = ray.pointAt(tSub);
                result.normal = gradientLocal(result.point);
                if (result.normal.length() > 0.0) {
                    result.normal.normalize();
                }

                return result;
            }

            t += std::max(newPhiAbs, _rayMarchingResolution);
            prevPhi = newPhi;
        }
    }

    return result;
}

template <size_t N>
Vector<double, N> CustomImplicitSurface<N>::gradientLocal(
    const Vector<double, N> &x) const {
    Vector<double, N> grad;
    for (size_t i = 0; i < N; ++i) {
        Vector<double, N> h;
        h[i] = 0.5 * _resolution;

        double lowerF = _func(x - h);
        double upperF = _func(x + h);

        grad[i] = (upperF - lowerF) / _resolution;
    }

    return grad;
}

template <size_t N>
typename CustomImplicitSurface<N>::Builder CustomImplicitSurface<N>::builder() {
    return Builder();
}

template <size_t N>
typename CustomImplicitSurface<N>::Builder &
CustomImplicitSurface<N>::Builder::withSignedDistanceFunction(
    const std::function<double(const Vector<double, N> &)> &func) {
    _func = func;
    return *this;
}

template <size_t N>
typename CustomImplicitSurface<N>::Builder &
CustomImplicitSurface<N>::Builder::withDomain(
    const BoundingBox<double, N> &domain) {
    _domain = domain;
    return *this;
}

template <size_t N>
typename CustomImplicitSurface<N>::Builder &
CustomImplicitSurface<N>::Builder::withResolution(double resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
typename CustomImplicitSurface<N>::Builder &
CustomImplicitSurface<N>::Builder::withRayMarchingResolution(
    double resolution) {
    _rayMarchingResolution = resolution;
    return *this;
}

template <size_t N>
typename CustomImplicitSurface<N>::Builder &
CustomImplicitSurface<N>::Builder::withMaxNumberOfIterations(
    unsigned int numIter) {
    _maxNumOfIterations = numIter;
    return *this;
}

template <size_t N>
CustomImplicitSurface<N> CustomImplicitSurface<N>::Builder::build() const {
    return CustomImplicitSurface(_func, _domain, _resolution,
                                 _rayMarchingResolution, _maxNumOfIterations,
                                 _transform, _isNormalFlipped);
}

template <size_t N>
std::shared_ptr<CustomImplicitSurface<N>>
CustomImplicitSurface<N>::Builder::makeShared() const {
    return std::shared_ptr<CustomImplicitSurface>(
        new CustomImplicitSurface(_func, _domain, _resolution,
                                  _rayMarchingResolution, _maxNumOfIterations,
                                  _transform, _isNormalFlipped),
        [](CustomImplicitSurface *obj) { delete obj; });
}

template class CustomImplicitSurface<2>;

template class CustomImplicitSurface<3>;

}  // namespace jet
