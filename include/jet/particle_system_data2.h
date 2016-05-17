// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PARTICLE_SYSTEM_DATA2_H_
#define INCLUDE_JET_PARTICLE_SYSTEM_DATA2_H_

#include <jet/array1.h>
#include <jet/point_neighbor_searcher2.h>

#include <memory>
#include <vector>

namespace jet {

class ParticleSystemData2 {
 public:
    typedef Array1<double> ScalarData;
    typedef Array1<Vector2D> VectorData;

    ParticleSystemData2();

    virtual ~ParticleSystemData2();

    void resize(size_t newNumberOfParticles);

    size_t numberOfParticles() const;

    size_t addScalarData(double initialVal = 0.0);

    size_t addVectorData(const Vector2D& initialVal = Vector2D());

    double radius() const;

    virtual void setRadius(double newRadius);

    double mass() const;

    virtual void setMass(double newMass);

    ConstArrayAccessor1<Vector2D> positions() const;

    ArrayAccessor1<Vector2D> positions();

    ConstArrayAccessor1<Vector2D> velocities() const;

    ArrayAccessor1<Vector2D> velocities();

    ConstArrayAccessor1<Vector2D> forces() const;

    ArrayAccessor1<Vector2D> forces();

    ConstArrayAccessor1<double> scalarDataAt(size_t idx) const;

    ArrayAccessor1<double> scalarDataAt(size_t idx);

    ConstArrayAccessor1<Vector2D> vectorDataAt(size_t idx) const;

    ArrayAccessor1<Vector2D> vectorDataAt(size_t idx);

    void addParticle(
        const Vector2D& newPosition,
        const Vector2D& newVelocity = Vector2D(),
        const Vector2D& newForce = Vector2D());

    //! Adds particles with given positions, velocities, and forces arrays.
    //! The size of the newVelocities and newForces must be the same as
    //! newPositions. Otherwise, std::invalid_argument will be thrown.
    void addParticles(
        const ConstArrayAccessor1<Vector2D>& newPositions,
        const ConstArrayAccessor1<Vector2D>& newVelocities
            = ConstArrayAccessor1<Vector2D>(),
        const ConstArrayAccessor1<Vector2D>& newForces
            = ConstArrayAccessor1<Vector2D>());

    const PointNeighborSearcher2Ptr& neighborSearcher() const;

    void setNeighborSearcher(
        const PointNeighborSearcher2Ptr& newNeighborSearcher);

    const std::vector<std::vector<size_t>>& neighborLists() const;

    void buildNeighborSearcher(double maxSearchRadius);

    void buildNeighborLists(double maxSearchRadius);

 private:
    double _radius = 1e-3;
    double _mass = 1e-3;
    VectorData _positions;
    VectorData _velocities;
    VectorData _forces;

    std::vector<ScalarData> _scalarDataList;
    std::vector<VectorData> _vectorDataList;

    PointNeighborSearcher2Ptr _neighborSearcher;
    std::vector<std::vector<size_t>> _neighborLists;
};

typedef std::shared_ptr<ParticleSystemData2> ParticleSystemData2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_SYSTEM_DATA2_H_
