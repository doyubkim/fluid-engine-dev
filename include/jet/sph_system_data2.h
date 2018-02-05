// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SPH_SYSTEM_DATA2_H_
#define INCLUDE_JET_SPH_SYSTEM_DATA2_H_

#include <jet/constants.h>
#include <jet/particle_system_data2.h>
#include <vector>

namespace jet {

//!
//! \brief      2-D SPH particle system data.
//!
//! This class extends ParticleSystemData2 to specialize the data model for SPH.
//! It includes density and pressure array as a default particle attribute, and
//! it also contains SPH utilities such as interpolation operator.
//!
class SphSystemData2 : public ParticleSystemData2 {
 public:
    //! Constructs empty SPH system.
    SphSystemData2();

    //! Constructs SPH system data with given number of particles.
    explicit SphSystemData2(size_t numberOfParticles);

    //! Copy constructor.
    SphSystemData2(const SphSystemData2& other);

    //! Destructor.
    virtual ~SphSystemData2();

    //!
    //! \brief      Sets the radius.
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

    //!
    //! \brief Updates the density array with the latest particle positions.
    //!
    //! This function updates the density array by recalculating each particle's
    //! latest nearby particles' position.
    //!
    //! \warning You must update the neighbor searcher
    //! (SphSystemData2::buildNeighborSearcher) before calling this function.
    //!
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
    //! \brief Returns the relative kernel radius.
    //!
    //! Returns the relative kernel radius compared to the target particle
    //! spacing (i.e. kernel radius / target spacing).
    //!
    double relativeKernelRadius() const;

    //!
    //! \brief Sets the absolute kernel radius.
    //!
    //! Sets the absolute kernel radius compared to the target particle
    //! spacing (i.e. relative kernel radius * target spacing).
    //! Once this function is called, hash grid and density should
    //! be updated using updateHashGrid() and updateDensities).
    //!
    void setKernelRadius(double kernelRadius);

    //! Returns the kernel radius in meters unit.
    double kernelRadius() const;

    //! Returns sum of kernel function evaluation for each nearby particle.
    double sumOfKernelNearby(const Vector2D& position) const;

    //!
    //! \brief Returns interpolated value at given origin point.
    //!
    //! Returns interpolated scalar data from the given position using
    //! standard SPH weighted average. The data array should match the
    //! particle layout. For example, density or pressure arrays can be
    //! used.
    //!
    //! \warning You must update the neighbor searcher
    //! (SphSystemData2::buildNeighborSearcher) before calling this function.
    //!
    double interpolate(const Vector2D& origin,
                       const ConstArrayAccessor1<double>& values) const;

    //!
    //! \brief Returns interpolated vector value at given origin point.
    //!
    //! Returns interpolated vector data from the given position using
    //! standard SPH weighted average. The data array should match the
    //! particle layout. For example, velocity or acceleration arrays can be
    //! used.
    //!
    //! \warning You must update the neighbor searcher
    //! (SphSystemData2::buildNeighborSearcher) before calling this function.
    //!
    Vector2D interpolate(const Vector2D& origin,
                         const ConstArrayAccessor1<Vector2D>& values) const;

    //!
    //! Returns the gradient of the given values at i-th particle.
    //!
    //! \warning You must update the neighbor lists
    //! (SphSystemData2::buildNeighborLists) before calling this function.
    //!
    Vector2D gradientAt(size_t i,
                        const ConstArrayAccessor1<double>& values) const;

    //!
    //! Returns the laplacian of the given values at i-th particle.
    //!
    //! \warning You must update the neighbor lists
    //! (SphSystemData2::buildNeighborLists) before calling this function.
    //!
    double laplacianAt(size_t i,
                       const ConstArrayAccessor1<double>& values) const;

    //!
    //! Returns the laplacian of the given values at i-th particle.
    //!
    //! \warning You must update the neighbor lists
    //! (SphSystemData2::buildNeighborLists) before calling this function.
    //!
    Vector2D laplacianAt(size_t i,
                         const ConstArrayAccessor1<Vector2D>& values) const;

    //! Builds neighbor searcher with kernel radius.
    void buildNeighborSearcher();

    //! Builds neighbor lists with kernel radius.
    void buildNeighborLists();

    //! Serializes this SPH system data to the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes this SPH system data from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Copies from other SPH system data.
    void set(const SphSystemData2& other);

    //! Copies from other SPH system data.
    SphSystemData2& operator=(const SphSystemData2& other);

 private:
    //! Target density of this particle system in kg/m^2.
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

//! Shared pointer for the SphSystemData2 type.
typedef std::shared_ptr<SphSystemData2> SphSystemData2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SPH_SYSTEM_DATA2_H_
