// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PARTICLE_SYSTEM_DATA_H_
#define INCLUDE_JET_PARTICLE_SYSTEM_DATA_H_

#include <jet/array.h>
#include <jet/point_neighbor_searcher.h>
#include <jet/serialization.h>

#ifndef JET_DOXYGEN

namespace flatbuffers {

class FlatBufferBuilder;
template <typename T>
struct Offset;

}  // namespace flatbuffers

namespace jet {
namespace fbs {

struct ParticleSystemData2;
struct ParticleSystemData3;

}  // namespace fbs
}  // namespace jet

#endif  // JET_DOXYGEN

namespace jet {

template <size_t N>
struct GetFlatbuffersParticleSystemData {};

template <>
struct GetFlatbuffersParticleSystemData<2> {
    using offset = flatbuffers::Offset<fbs::ParticleSystemData2>;

    static const fbs::ParticleSystemData2* getParticleSystemData(
        const uint8_t* data);
};

template <>
struct GetFlatbuffersParticleSystemData<3> {
    using offset = flatbuffers::Offset<fbs::ParticleSystemData3>;

    static const fbs::ParticleSystemData3* getParticleSystemData(
        const uint8_t* data);
};

//!
//! \brief      N-D particle system data.
//!
//! This class is the key data structure for storing particle system data. A
//! single particle has position, velocity, and force attributes by default. But
//! it can also have additional custom scalar or vector attributes.
//!
template <size_t N>
class ParticleSystemData : public Serializable {
 public:
    //! Scalar data chunk.
    typedef Array1<double> ScalarData;

    //! Vector data chunk.
    typedef Array1<Vector<double, N>> VectorData;

    //! Default constructor.
    ParticleSystemData();

    //! Constructs particle system data with given number of particles.
    explicit ParticleSystemData(size_t numberOfParticles);

    //! Copy constructor.
    ParticleSystemData(const ParticleSystemData& other);

    //! Destructor.
    virtual ~ParticleSystemData();

    //!
    //! \brief      Resizes the number of particles of the container.
    //!
    //! This function will resize internal containers to store newly given
    //! number of particles including custom data layers. However, this will
    //! invalidate neighbor searcher and neighbor lists. It is users
    //! responsibility to call ParticleSystemData::buildNeighborSearcher and
    //! ParticleSystemData::buildNeighborLists to refresh those data.
    //!
    //! \param[in]  newNumberOfParticles    New number of particles.
    //!
    void resize(size_t newNumberOfParticles);

    //! Returns the number of particles.
    size_t numberOfParticles() const;

    //!
    //! \brief      Adds a scalar data layer and returns its index.
    //!
    //! This function adds a new scalar data layer to the system. It can be used
    //! for adding a scalar attribute, such as temperature, to the particles.
    //!
    //! \params[in] initialVal  Initial value of the new scalar data.
    //!
    size_t addScalarData(double initialVal = 0.0);

    //!
    //! \brief      Adds a vector data layer and returns its index.
    //!
    //! This function adds a new vector data layer to the system. It can be used
    //! for adding a vector attribute, such as vortex, to the particles.
    //!
    //! \params[in] initialVal  Initial value of the new vector data.
    //!
    size_t addVectorData(
        const Vector<double, N>& initialVal = Vector<double, N>());

    //! Returns the radius of the particles.
    double radius() const;

    //! Sets the radius of the particles.
    virtual void setRadius(double newRadius);

    //! Returns the mass of the particles.
    double mass() const;

    //! Sets the mass of the particles.
    virtual void setMass(double newMass);

    //! Returns the position array (immutable).
    ConstArrayView1<Vector<double, N>> positions() const;

    //! Returns the position array (mutable).
    ArrayView1<Vector<double, N>> positions();

    //! Returns the velocity array (immutable).
    ConstArrayView1<Vector<double, N>> velocities() const;

    //! Returns the velocity array (mutable).
    ArrayView1<Vector<double, N>> velocities();

    //! Returns the force array (immutable).
    ConstArrayView1<Vector<double, N>> forces() const;

    //! Returns the force array (mutable).
    ArrayView1<Vector<double, N>> forces();

    //! Returns custom scalar data layer at given index (immutable).
    ConstArrayView1<double> scalarDataAt(size_t idx) const;

    //! Returns custom scalar data layer at given index (mutable).
    ArrayView1<double> scalarDataAt(size_t idx);

    //! Returns custom vector data layer at given index (immutable).
    ConstArrayView1<Vector<double, N>> vectorDataAt(size_t idx) const;

    //! Returns custom vector data layer at given index (mutable).
    ArrayView1<Vector<double, N>> vectorDataAt(size_t idx);

    //!
    //! \brief      Adds a particle to the data structure.
    //!
    //! This function will add a single particle to the data structure. For
    //! custom data layers, zeros will be assigned for new particles.
    //! However, this will invalidate neighbor searcher and neighbor lists. It
    //! is users responsibility to call
    //! ParticleSystemData::buildNeighborSearcher and
    //! ParticleSystemData::buildNeighborLists to refresh those data.
    //!
    //! \param[in]  newPosition The new position.
    //! \param[in]  newVelocity The new velocity.
    //! \param[in]  newForce    The new force.
    //!
    void addParticle(const Vector<double, N>& newPosition,
                     const Vector<double, N>& newVelocity = Vector<double, N>(),
                     const Vector<double, N>& newForce = Vector<double, N>());

    //!
    //! \brief      Adds particles to the data structure.
    //!
    //! This function will add particles to the data structure. For custom data
    //! layers, zeros will be assigned for new particles. However, this will
    //! invalidate neighbor searcher and neighbor lists. It is users
    //! responsibility to call ParticleSystemData::buildNeighborSearcher and
    //! ParticleSystemData::buildNeighborLists to refresh those data.
    //!
    //! \param[in]  newPositions  The new positions.
    //! \param[in]  newVelocities The new velocities.
    //! \param[in]  newForces     The new forces.
    //!
    void addParticles(const ConstArrayView1<Vector<double, N>>& newPositions,
                      const ConstArrayView1<Vector<double, N>>& newVelocities =
                          ConstArrayView1<Vector<double, N>>(),
                      const ConstArrayView1<Vector<double, N>>& newForces =
                          ConstArrayView1<Vector<double, N>>());

    //!
    //! \brief      Returns neighbor searcher.
    //!
    //! This function returns currently set neighbor searcher object. By
    //! default, PointParallelHashGridSearcher2 is used.
    //!
    //! \return     Current neighbor searcher.
    //!
    const std::shared_ptr<PointNeighborSearcher<N>>& neighborSearcher() const;

    //! Sets neighbor searcher.
    void setNeighborSearcher(
        const std::shared_ptr<PointNeighborSearcher<N>>& newNeighborSearcher);

    //!
    //! \brief      Returns neighbor lists.
    //!
    //! This function returns neighbor lists which is available after calling
    //! PointParallelHashGridSearcher2::buildNeighborLists. Each list stores
    //! indices of the neighbors.
    //!
    //! \return     Neighbor lists.
    //!
    const Array1<Array1<size_t>>& neighborLists() const;

    //! Builds neighbor searcher with given search radius.
    void buildNeighborSearcher(double maxSearchRadius);

    //! Builds neighbor lists with given search radius.
    void buildNeighborLists(double maxSearchRadius);

    //! Serializes this particle system data to the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes this particle system data from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Copies from other particle system data.
    void set(const ParticleSystemData& other);

    //! Copies from other particle system data.
    ParticleSystemData& operator=(const ParticleSystemData& other);

 protected:
    template <size_t M = N>
    static std::enable_if_t<M == 2, void> serialize(
        const ParticleSystemData<2>& particles,
        flatbuffers::FlatBufferBuilder* builder,
        flatbuffers::Offset<fbs::ParticleSystemData2>* fbsParticleSystemData);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> serialize(
        const ParticleSystemData<3>& particles,
        flatbuffers::FlatBufferBuilder* builder,
        flatbuffers::Offset<fbs::ParticleSystemData3>* fbsParticleSystemData);

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> deserialize(
        const fbs::ParticleSystemData2* fbsParticleSystemData,
        ParticleSystemData<2>& particles);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> deserialize(
        const fbs::ParticleSystemData3* fbsParticleSystemData,
        ParticleSystemData<3>& particles);

 private:
    double _radius = 1e-3;
    double _mass = 1e-3;
    size_t _numberOfParticles = 0;
    size_t _positionIdx;
    size_t _velocityIdx;
    size_t _forceIdx;

    Array1<ScalarData> _scalarDataList;
    Array1<VectorData> _vectorDataList;

    std::shared_ptr<PointNeighborSearcher<N>> _neighborSearcher;
    Array1<Array1<size_t>> _neighborLists;
};

//! 2-D ParticleSystemData type.
using ParticleSystemData2 = ParticleSystemData<2>;

//! 3-D ParticleSystemData type.
using ParticleSystemData3 = ParticleSystemData<3>;

//! Shared pointer type of ParticleSystemData2.
using ParticleSystemData2Ptr = std::shared_ptr<ParticleSystemData2>;

//! Shared pointer type of ParticleSystemData3.
using ParticleSystemData3Ptr = std::shared_ptr<ParticleSystemData3>;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_SYSTEM_DATA2_H_
