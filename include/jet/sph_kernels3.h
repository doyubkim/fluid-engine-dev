// Copyright (c) 2016 Doyub Kim
//
// Adopted from the sample code of:
// Bart Adams and Martin Wicke,
// "Meshless Approximation Methods and Applications in Physics Based Modeling
// and Animation", Eurographics 2009 Tutorial

#ifndef INCLUDE_JET_SPH_KERNELS3_H_
#define INCLUDE_JET_SPH_KERNELS3_H_

#include <jet/constants.h>
#include <jet/vector3.h>

namespace jet {

//!
//! \brief Standard 3-D SPH kernel function object.
//!
struct SphStdKernel3 {
    double h, h2, h3, h5;

    SphStdKernel3();

    explicit SphStdKernel3(double kernelRadius);

    SphStdKernel3(const SphStdKernel3& other);

    double operator()(double distance) const;

    double firstDerivative(double distance) const;

    Vector3D gradient(double distance, const Vector3D& direction) const;

    double secondDerivative(double distance) const;
};

//!
//! \brief Spiky 3-D SPH kernel function object.
//!
struct SphSpikyKernel3 {
    double h, h2, h3, h4, h5;

    SphSpikyKernel3();

    explicit SphSpikyKernel3(double kernelRadius);

    SphSpikyKernel3(const SphSpikyKernel3& other);

    double operator()(double distance) const;

    double firstDerivative(double distance) const;

    Vector3D gradient(double distance, const Vector3D& direction) const;

    double secondDerivative(double distance) const;
};

}  // namespace jet

#include "detail/sph_kernels3-inl.h"

#endif  // INCLUDE_JET_SPH_KERNELS3_H_
