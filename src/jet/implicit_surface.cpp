// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/implicit_surface.h>

namespace jet {

template <size_t N>
ImplicitSurface<N>::ImplicitSurface(const Transform<N> &transform,
                                    bool isNormalFlipped)
    : Surface<N>(transform, isNormalFlipped) {}

template <size_t N>
ImplicitSurface<N>::ImplicitSurface(const ImplicitSurface &other)
    : Surface<N>(other) {}

template <size_t N>
ImplicitSurface<N>::~ImplicitSurface() {}

template <size_t N>
double ImplicitSurface<N>::signedDistance(
    const Vector<double, N> &otherPoint) const {
    return signedDistanceLocal(transform.toLocal(otherPoint));
}

template <size_t N>
double ImplicitSurface<N>::closestDistanceLocal(
    const Vector<double, N> &otherPoint) const {
    return std::fabs(signedDistanceLocal(otherPoint));
}

template class ImplicitSurface<2>;

template class ImplicitSurface<3>;

}  // namespace jet
