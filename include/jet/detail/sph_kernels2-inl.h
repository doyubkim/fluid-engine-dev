// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// Adopted from the sample code of:
// Bart Adams and Martin Wicke,
// "Meshless Approximation Methods and Applications in Physics Based Modeling
// and Animation", Eurographics 2009 Tutorial

#ifndef INCLUDE_JET_DETAIL_SPH_KERNELS2_INL_H_
#define INCLUDE_JET_DETAIL_SPH_KERNELS2_INL_H_

#include <jet/constants.h>

namespace jet {

inline SphStdKernel2::SphStdKernel2()
    : h(0), h2(0), h3(0), h4(0) {}

inline SphStdKernel2::SphStdKernel2(double h_)
    : h(h_), h2(h * h), h3(h2 * h), h4(h2 * h2) {}

inline SphStdKernel2::SphStdKernel2(const SphStdKernel2& other)
    : h(other.h), h2(other.h2), h3(other.h3), h4(other.h4) {}

inline double SphStdKernel2::operator()(double distance) const {
    double distanceSquared = distance * distance;

    if (distanceSquared >= h2) {
        return 0.0;
    } else {
        double x = 1.0 - distanceSquared / h2;
        return 4.0 / (kPiD * h2) * x * x * x;
    }
}

inline double SphStdKernel2::firstDerivative(double distance) const {
    if (distance >= h) {
        return 0.0;
    } else {
        double x = 1.0 - distance * distance / h2;
        return -24.0 * distance / (kPiD * h4) * x * x;
    }
}

inline Vector2D SphStdKernel2::gradient(
    const Vector2D& point) const {
    double dist = point.length();
    if (dist > 0.0) {
        return gradient(dist, point / dist);
    } else {
        return Vector2D(0, 0);
    }
}

inline Vector2D SphStdKernel2::gradient(
    double distance,
    const Vector2D& directionToCenter) const {
    return -firstDerivative(distance) * directionToCenter;
}

inline double SphStdKernel2::secondDerivative(double distance) const {
    double distanceSquared = distance * distance;

    if (distanceSquared >= h2) {
        return 0.0;
    } else {
        double x = distanceSquared / h2;
        return 24.0 / (kPiD * h4) * (1 - x) * (5 * x - 1);
    }
}


inline SphSpikyKernel2::SphSpikyKernel2()
    : h(0), h2(0), h3(0), h4(0), h5(0) {}

inline SphSpikyKernel2::SphSpikyKernel2(double h_)
    : h(h_), h2(h * h), h3(h2 * h), h4(h2 * h2), h5(h3 * h2) {}

inline SphSpikyKernel2::SphSpikyKernel2(const SphSpikyKernel2& other)
    : h(other.h), h2(other.h2), h3(other.h3), h4(other.h4), h5(other.h5) {}

inline double SphSpikyKernel2::operator()(double distance) const {
    if (distance >= h) {
        return 0.0;
    } else {
        double x = 1.0 - distance / h;
        return 10.0 / (kPiD * h2) * x * x * x;
    }
}

inline double SphSpikyKernel2::firstDerivative(double distance) const {
    if (distance >= h) {
        return 0.0;
    } else {
        double x = 1.0 - distance / h;
        return -30.0 / (kPiD * h3) * x * x;
    }
}

inline Vector2D SphSpikyKernel2::gradient(
    const Vector2D& point) const {
    double dist = point.length();
    if (dist > 0.0) {
        return gradient(dist, point / dist);
    } else {
        return Vector2D(0, 0);
    }
}

inline Vector2D SphSpikyKernel2::gradient(
    double distance,
    const Vector2D& directionToCenter) const {
    return -firstDerivative(distance) * directionToCenter;
}

inline double SphSpikyKernel2::secondDerivative(double distance) const {
    if (distance >= h) {
        return 0.0;
    } else {
        double x = 1.0 - distance / h;
        return 60.0 / (kPiD * h4) * x;
    }
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SPH_KERNELS2_INL_H_
