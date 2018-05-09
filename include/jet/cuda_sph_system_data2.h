// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_SPH_SYSTEM_DATA2_H_
#define INCLUDE_JET_CUDA_SPH_SYSTEM_DATA2_H_

#include <jet/constants.h>
#include <jet/cuda_particle_system_data2.h>

namespace jet {

namespace experimental {

class CudaSphSystemData2 : public CudaParticleSystemData2 {
 public:
    CudaSphSystemData2();

    explicit CudaSphSystemData2(size_t numberOfParticles);

    CudaSphSystemData2(const CudaSphSystemData2& other);

    virtual ~CudaSphSystemData2();

    CudaArrayView1<float> densities();

    ConstCudaArrayView1<float> densities() const;

    CudaArrayView1<float> pressures();

    ConstCudaArrayView1<float> pressures() const;

    void updateDensities();

    //! Returns the target density of this particle system.
    float targetDensity() const;

    //! Sets the target density of this particle system.
    void setTargetDensity(float targetDensity);

    //! Returns the target particle spacing in meters.
    float targetSpacing() const;

    //!
    //! \brief Sets the target particle spacing in meters.
    //!
    //! Once this function is called, hash grid and density should be
    //! updated using updateHashGrid() and updateDensities).
    //!
    void setTargetSpacing(float spacing);

    //!
    //! \brief Returns the relative kernel radius.
    //!
    //! Returns the relative kernel radius compared to the target particle
    //! spacing (i.e. kernel radius / target spacing).
    //!
    float relativeKernelRadius() const;

    //!
    //! \brief Sets the relative kernel radius.
    //!
    //! Sets the relative kernel radius compared to the target particle
    //! spacing (i.e. kernel radius / target spacing).
    //! Once this function is called, hash grid and density should
    //! be updated using updateHashGrid() and updateDensities).
    //!
    void setRelativeKernelRadius(float relativeRadius);

    //! Returns the kernel radius in meters unit.
    float kernelRadius() const;

    //!
    //! \brief Sets the absolute kernel radius.
    //!
    //! Sets the absolute kernel radius compared to the target particle
    //! spacing (i.e. relative kernel radius * target spacing).
    //! Once this function is called, hash grid and density should
    //! be updated using updateHashGrid() and updateDensities).
    //!
    void setKernelRadius(float kernelRadius);

    float mass() const;

    //! Builds neighbor searcher with kernel radius.
    void buildNeighborSearcher();

    //! Builds neighbor lists with kernel radius and update densities.
    void buildNeighborListsAndUpdateDensities();

    //! Copies from other SPH system data.
    void set(const CudaSphSystemData2& other);

    //! Copies from other SPH system data.
    CudaSphSystemData2& operator=(const CudaSphSystemData2& other);

 private:
    //! Target density of this particle system in kg/m^2.
    float _targetDensity = kWaterDensityF;

    //! Target spacing of this particle system in meters.
    float _targetSpacing = 0.1f;

    //! Relative radius of SPH kernel.
    //! SPH kernel radius divided by target spacing.
    float _kernelRadiusOverTargetSpacing = 1.8f;

    //! SPH kernel radius in meters.
    float _kernelRadius;

    float _mass;

    size_t _pressureIdx;

    size_t _densityIdx;

    //! Computes the mass based on the target density and spacing.
    void computeMass();
};

//! Shared pointer for the CudaSphSystemData2 type.
typedef std::shared_ptr<CudaSphSystemData2> CudaSphSystemData2Ptr;

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_SPH_SYSTEM_DATA2_H_

#endif  // JET_USE_CUDA
