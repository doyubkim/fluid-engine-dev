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

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_SPH_KERNELS2_H_
#define INCLUDE_JET_CUDA_SPH_KERNELS2_H_

#include <jet/constants.h>

namespace jet {

//!
//! \brief Standard 2-D SPH kernel function object.
//!
//! \see Müller, Matthias, David Charypar, and Markus Gross.
//!     "Particle-based fluid simulation for interactive applications."
//!     Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer
//!     animation. Eurographics Association, 2003.
//!
struct CudaSphStdKernel2 {
    //! Kernel radius.
    float h;

    //! Square of the kernel radius.
    float h2;

    //! Cubic of the kernel radius.
    float h3;

    //! Forth-power of the kernel radius.
    float h4;

    //! Constructs a kernel object with zero radius.
    JET_CUDA_HOST_DEVICE CudaSphStdKernel2();

    //! Constructs a kernel object with given radius.
    JET_CUDA_HOST_DEVICE explicit CudaSphStdKernel2(float kernelRadius);

    //! Copy constructor
    JET_CUDA_HOST_DEVICE CudaSphStdKernel2(const CudaSphStdKernel2& other);

    //! Returns kernel function value at given distance.
    JET_CUDA_HOST_DEVICE float operator()(float distance) const;

    //! Returns the first derivative at given distance.
    JET_CUDA_HOST_DEVICE float firstDerivative(float distance) const;

    //! Returns the gradient at a point.
    JET_CUDA_HOST_DEVICE float2 gradient(const float2& point) const;

    //! Returns the gradient at a point defined by distance and direction.
    JET_CUDA_HOST_DEVICE float2 gradient(float distance,
                                         const float2& direction) const;

    //! Returns the second derivative at given distance.
    JET_CUDA_HOST_DEVICE float secondDerivative(float distance) const;
};

//!
//! \brief Spiky 2-D SPH kernel function object.
//!
//! \see Müller, Matthias, David Charypar, and Markus Gross.
//!     "Particle-based fluid simulation for interactive applications."
//!     Proceedings of the 2002 ACM SIGGRAPH/Eurographics symposium on Computer
//!     animation. Eurographics Association, 2002.
//!
struct CudaSphSpikyKernel2 {
    //! Kernel radius.
    float h;

    //! Square of the kernel radius.
    float h2;

    //! Cubic of the kernel radius.
    float h3;

    //! Fourth-power of the kernel radius.
    float h4;

    //! Fifth-power of the kernel radius.
    float h5;

    //! Constructs a kernel object with zero radius.
    JET_CUDA_HOST_DEVICE CudaSphSpikyKernel2();

    //! Constructs a kernel object with given radius.
    JET_CUDA_HOST_DEVICE explicit CudaSphSpikyKernel2(float kernelRadius);

    //! Copy constructor
    JET_CUDA_HOST_DEVICE CudaSphSpikyKernel2(const CudaSphSpikyKernel2& other);

    //! Returns kernel function value at given distance.
    JET_CUDA_HOST_DEVICE float operator()(float distance) const;

    //! Returns the first derivative at given distance.
    JET_CUDA_HOST_DEVICE float firstDerivative(float distance) const;

    //! Returns the gradient at a point.
    JET_CUDA_HOST_DEVICE float2 gradient(const float2& point) const;

    //! Returns the gradient at a point defined by distance and direction.
    JET_CUDA_HOST_DEVICE float2 gradient(float distance,
                                         const float2& direction) const;

    //! Returns the second derivative at given distance.
    JET_CUDA_HOST_DEVICE float secondDerivative(float distance) const;
};

}  // namespace jet

#include "detail/cuda_sph_kernels2-inl.h"

#endif  // INCLUDE_JET_CUDA_SPH_KERNELS2_H_

#endif  // JET_USE_CUDA
