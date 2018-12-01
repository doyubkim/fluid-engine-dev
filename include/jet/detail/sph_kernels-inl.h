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

#ifndef INCLUDE_JET_DETAIL_SPH_KERNELS_INL_H_
#define INCLUDE_JET_DETAIL_SPH_KERNELS_INL_H_

#include <jet/sph_kernels.h>
#include <jet/constants.h>

namespace jet {

// MARK: SphStdKernel2 implementations

inline SphStdKernel2::SphStdKernel() : h(0), h2(0), h3(0), h4(0) {}

inline SphStdKernel2::SphStdKernel(double h_)
    : h(h_), h2(h * h), h3(h2 * h), h4(h2 * h2) {}

inline SphStdKernel2::SphStdKernel(const SphStdKernel& other)
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

inline Vector2D SphStdKernel2::gradient(const Vector2D& point) const {
    double dist = point.length();
    if (dist > 0.0) {
        return gradient(dist, point / dist);
    } else {
        return Vector2D(0, 0);
    }
}

inline Vector2D SphStdKernel2::gradient(
    double distance, const Vector2D& directionToCenter) const {
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

// MARK: SphSpikyKernel2 implementations

inline SphSpikyKernel2::SphSpikyKernel() : h(0), h2(0), h3(0), h4(0), h5(0) {}

inline SphSpikyKernel2::SphSpikyKernel(double h_)
    : h(h_), h2(h * h), h3(h2 * h), h4(h2 * h2), h5(h3 * h2) {}

inline SphSpikyKernel2::SphSpikyKernel(const SphSpikyKernel2& other)
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

inline Vector2D SphSpikyKernel2::gradient(const Vector2D& point) const {
    double dist = point.length();
    if (dist > 0.0) {
        return gradient(dist, point / dist);
    } else {
        return Vector2D(0, 0);
    }
}

inline Vector2D SphSpikyKernel2::gradient(
    double distance, const Vector2D& directionToCenter) const {
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

// MARK: SphStdKernel3 implementations

inline SphStdKernel3::SphStdKernel() : h(0), h2(0), h3(0), h5(0) {}

inline SphStdKernel3::SphStdKernel(double kernelRadius)
    : h(kernelRadius), h2(h * h), h3(h2 * h), h5(h2 * h3) {}

inline SphStdKernel3::SphStdKernel(const SphStdKernel& other)
    : h(other.h), h2(other.h2), h3(other.h3), h5(other.h5) {}

inline double SphStdKernel3::operator()(double distance) const {
    if (distance * distance >= h2) {
        return 0.0;
    } else {
        double x = 1.0 - distance * distance / h2;
        return 315.0 / (64.0 * kPiD * h3) * x * x * x;
    }
}

inline double SphStdKernel3::firstDerivative(double distance) const {
    if (distance >= h) {
        return 0.0;
    } else {
        double x = 1.0 - distance * distance / h2;
        return -945.0 / (32.0 * kPiD * h5) * distance * x * x;
    }
}

inline Vector3D SphStdKernel3::gradient(const Vector3D& point) const {
    double dist = point.length();
    if (dist > 0.0) {
        return gradient(dist, point / dist);
    } else {
        return Vector3D(0, 0, 0);
    }
}

inline Vector3D SphStdKernel3::gradient(
    double distance, const Vector3D& directionToCenter) const {
    return -firstDerivative(distance) * directionToCenter;
}

inline double SphStdKernel3::secondDerivative(double distance) const {
    if (distance * distance >= h2) {
        return 0.0;
    } else {
        double x = distance * distance / h2;
        return 945.0 / (32.0 * kPiD * h5) * (1 - x) * (3 * x - 1);
    }
}

// MARK: SphSpikyKernel3 implementations

inline SphSpikyKernel3::SphSpikyKernel() : h(0), h2(0), h3(0), h4(0), h5(0) {}

inline SphSpikyKernel3::SphSpikyKernel(double h_)
    : h(h_), h2(h * h), h3(h2 * h), h4(h2 * h2), h5(h3 * h2) {}

inline SphSpikyKernel3::SphSpikyKernel(const SphSpikyKernel3& other)
    : h(other.h), h2(other.h2), h3(other.h3), h4(other.h4), h5(other.h5) {}

inline double SphSpikyKernel3::operator()(double distance) const {
    if (distance >= h) {
        return 0.0;
    } else {
        double x = 1.0 - distance / h;
        return 15.0 / (kPiD * h3) * x * x * x;
    }
}

inline double SphSpikyKernel3::firstDerivative(double distance) const {
    if (distance >= h) {
        return 0.0;
    } else {
        double x = 1.0 - distance / h;
        return -45.0 / (kPiD * h4) * x * x;
    }
}

inline Vector3D SphSpikyKernel3::gradient(const Vector3D& point) const {
    double dist = point.length();
    if (dist > 0.0) {
        return gradient(dist, point / dist);
    } else {
        return Vector3D(0, 0, 0);
    }
}

inline Vector3D SphSpikyKernel3::gradient(
    double distance, const Vector3D& directionToCenter) const {
    return -firstDerivative(distance) * directionToCenter;
}

inline double SphSpikyKernel3::secondDerivative(double distance) const {
    if (distance >= h) {
        return 0.0;
    } else {
        double x = 1.0 - distance / h;
        return 90.0 / (kPiD * h5) * x;
    }
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SPH_KERNELS_INL_H_
