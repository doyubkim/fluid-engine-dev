// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PARTICLE_SYSTEM_DATA3_H_
#define INCLUDE_JET_PARTICLE_SYSTEM_DATA3_H_

#include <jet/array1.h>
#include <jet/point_neighbor_searcher3.h>

#include <memory>
#include <vector>

namespace jet {

class ParticleSystemData3 {
 public:
    typedef Array1<double> ScalarData;
    typedef Array1<Vector3D> VectorData;

    ParticleSystemData3();

    virtual ~ParticleSystemData3();

    void resize(size_t newNumberOfParticles);

    size_t numberOfParticles() const;

    size_t addScalarData(double initialVal = 0.0);

    size_t addVectorData(const Vector3D& initialVal = Vector3D());

    double radius() const;

    virtual void setRadius(double newRadius);

    double mass() const;

    virtual void setMass(double newMass);

    ConstArrayAccessor1<Vector3D> positions() const;

    ArrayAccessor1<Vector3D> positions();

    ConstArrayAccessor1<Vector3D> velocities() const;

    ArrayAccessor1<Vector3D> velocities();

    ConstArrayAccessor1<Vector3D> forces() const;

    ArrayAccessor1<Vector3D> forces();

    ConstArrayAccessor1<double> scalarDataAt(size_t idx) const;

    ArrayAccessor1<double> scalarDataAt(size_t idx);

    ConstArrayAccessor1<Vector3D> vectorDataAt(size_t idx) const;

    ArrayAccessor1<Vector3D> vectorDataAt(size_t idx);

    void addParticle(
        const Vector3D& newPosition,
        const Vector3D& newVelocity = Vector3D(),
        const Vector3D& newForce = Vector3D());

    //! Adds particles with given positions, velocities, and forces arrays.
    //! The size of the newVelocities and newForces must be the same as
    //! newPositions. Otherwise, std::invalid_argument will be thrown.
    void addParticles(
        const ConstArrayAccessor1<Vector3D>& newPositions,
        const ConstArrayAccessor1<Vector3D>& newVelocities
            = ConstArrayAccessor1<Vector3D>(),
        const ConstArrayAccessor1<Vector3D>& newForces
            = ConstArrayAccessor1<Vector3D>());

    const PointNeighborSearcher3Ptr& neighborSearcher() const;

    void setNeighborSearcher(
        const PointNeighborSearcher3Ptr& newNeighborSearcher);

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

    PointNeighborSearcher3Ptr _neighborSearcher;
    std::vector<std::vector<size_t>> _neighborLists;
};

typedef std::shared_ptr<ParticleSystemData3> ParticleSystemData3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_SYSTEM_DATA3_H_
