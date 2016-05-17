// Copyright (c) 2016 Doyub Kim
//
// Adopted from the sample code of:
// Bart Adams and Martin Wicke,
// "Meshless Approximation Methods and Applications in Physics Based Modeling
// and Animation", Eurographics 2009 Tutorial

#ifndef INCLUDE_JET_SPH_KERNELS2_H_
#define INCLUDE_JET_SPH_KERNELS2_H_

#include <jet/constants.h>
#include <jet/vector2.h>

namespace jet {

//!
//! \brief Standard 2-D SPH kernel function object.
//!
struct SphStdKernel2 {
    double h, h2, h3, h4;

    SphStdKernel2();

    explicit SphStdKernel2(double kernelRadius);

    SphStdKernel2(const SphStdKernel2& other);

    double operator()(double distance) const;

    double firstDerivative(double distance) const;

    Vector2D gradient(double distance, const Vector2D& direction) const;

    double secondDerivative(double distance) const;
};

//!
//! \brief Spiky 2-D SPH kernel function object.
//!
struct SphSpikyKernel2 {
    double h, h2, h3, h4, h5;

    SphSpikyKernel2();

    explicit SphSpikyKernel2(double kernelRadius);

    SphSpikyKernel2(const SphSpikyKernel2& other);

    double operator()(double distance) const;

    double firstDerivative(double distance) const;

    Vector2D gradient(double distance, const Vector2D& direction) const;

    double secondDerivative(double distance) const;
};

}  // namespace jet

#include "detail/sph_kernels2-inl.h"

#endif  // INCLUDE_JET_SPH_KERNELS2_H_
