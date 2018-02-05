// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SPH_SYSTEM_DATA3_H_
#define INCLUDE_JET_SPH_SYSTEM_DATA3_H_

#include <jet/constants.h>
#include <jet/particle_system_data3.h>
#include <vector>

namespace jet {

//!
//! \brief      3-D SPH particle system data.
//!
//! This class extends ParticleSystemData3 to specialize the data model for SPH.
//! It includes density and pressure array as a default particle attribute, and
//! it also contains SPH utilities such as interpolation operator.
//!
class SphSystemData3 : public ParticleSystemData3 {
 public:
    //! Constructs empty SPH system.
    SphSystemData3();

    //! Constructs SPH system data with given number of particles.
    explicit SphSystemData3(size_t numberOfParticles);

    //! Copy constructor.
    SphSystemData3(const SphSystemData3& other);

    //! Destructor.
    virtual ~SphSystemData3();

    //!
    //! \brief Sets the radius.
    //!
    //! Sets the radius of the particle system. The radius will be interpreted
    //! as target spacing.
    //!
    void setRadius(double newRadius) override;

    //!
    //! \brief      Sets the mass of a particle.
    //!
    //! Setting the mass of a particle will change the target density.
    //!
    //! \param[in]  newMass The new mass.
    //!
    void setMass(double newMass) override;

    //! Returns the density array accessor (immutable).
    ConstArrayAccessor1<double> densities() const;

    //! Returns the density array accessor (mutable).
    ArrayAccessor1<double> densities();

    //! Returns the pressure array accessor (immutable).
    ConstArrayAccessor1<double> pressures() const;

    //! Returns the pressure array accessor (mutable).
    ArrayAccessor1<double> pressures();

    //! Updates the density array with the latest particle positions.
    void updateDensities();

    //! Sets the target density of this particle system.
    void setTargetDensity(double targetDensity);

    //! Returns the target density of this particle system.
    double targetDensity() const;

    //!
    //! \brief Sets the target particle spacing in meters.
    //!
    //! Once this function is called, hash grid and density should be
    //! updated using updateHashGrid() and updateDensities).
    //!
    void setTargetSpacing(double spacing);

    //! Returns the target particle spacing in meters.
    double targetSpacing() const;

    //!
    //! \brief Sets the relative kernel radius.
    //!
    //! Sets the relative kernel radius compared to the target particle
    //! spacing (i.e. kernel radius / target spacing).
    //! Once this function is called, hash grid and density should
    //! be updated using updateHashGrid() and updateDensities).
    //!
    void setRelativeKernelRadius(double relativeRadius);

    //!
    //! \brief Sets the absolute kernel radius.
    //!
    //! Sets the absolute kernel radius compared to the target particle
    //! spacing (i.e. relative kernel radius * target spacing).
    //! Once this function is called, hash grid and density should
    //! be updated using updateHashGrid() and updateDensities).
    //!
    void setKernelRadius(double kernelRadius);

    //!
    //! \brief Returns the relative kernel radius.
    //!
    //! Returns the relative kernel radius compared to the target particle
    //! spacing (i.e. kernel radius / target spacing).
    //!
    double relativeKernelRadius() const;

    //! Returns the kernel radius in meters unit.
    double kernelRadius() const;

    //! Returns sum of kernel function evaluation for each nearby particle.
    double sumOfKernelNearby(const Vector3D& position) const;

    //!
    //! \brief Returns interpolated value at given origin point.
    //!
    //! Returns interpolated scalar data from the given position using
    //! standard SPH weighted average. The data array should match the
    //! particle layout. For example, density or pressure arrays can be
    //! used.
    //!
    double interpolate(const Vector3D& origin,
                       const ConstArrayAccessor1<double>& values) const;

    //!
    //! \brief Returns interpolated vector value at given origin point.
    //!
    //! Returns interpolated vector data from the given position using
    //! standard SPH weighted average. The data array should match the
    //! particle layout. For example, velocity or acceleration arrays can be
    //! used.
    //!
    Vector3D interpolate(const Vector3D& origin,
                         const ConstArrayAccessor1<Vector3D>& values) const;

    //! Returns the gradient of the given values at i-th particle.
    Vector3D gradientAt(size_t i,
                        const ConstArrayAccessor1<double>& values) const;

    //! Returns the laplacian of the given values at i-th particle.
    double laplacianAt(size_t i,
                       const ConstArrayAccessor1<double>& values) const;

    //! Returns the laplacian of the given values at i-th particle.
    Vector3D laplacianAt(size_t i,
                         const ConstArrayAccessor1<Vector3D>& values) const;

    //! Builds neighbor searcher with kernel radius.
    void buildNeighborSearcher();

    //! Builds neighbor lists with kernel radius.
    void buildNeighborLists();

    //! Serializes this SPH system data to the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes this SPH system data from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Copies from other SPH system data.
    void set(const SphSystemData3& other);

    //! Copies from other SPH system data.
    SphSystemData3& operator=(const SphSystemData3& other);

 private:
    //! Target density of this particle system in kg/m^3.
    double _targetDensity = kWaterDensity;

    //! Target spacing of this particle system in meters.
    double _targetSpacing = 0.1;

    //! Relative radius of SPH kernel.
    //! SPH kernel radius divided by target spacing.
    double _kernelRadiusOverTargetSpacing = 1.8;

    //! SPH kernel radius in meters.
    double _kernelRadius;

    size_t _pressureIdx;

    size_t _densityIdx;

    //! Computes the mass based on the target density and spacing.
    void computeMass();
};

//! Shared pointer for the SphSystemData3 type.
typedef std::shared_ptr<SphSystemData3> SphSystemData3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SPH_SYSTEM_DATA3_H_
