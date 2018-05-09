// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_PARTICLE_SYSTEM_DATA2_H_
#define INCLUDE_JET_CUDA_PARTICLE_SYSTEM_DATA2_H_

#include <jet/array_view1.h>
#include <jet/cuda_array1.h>
#include <jet/cuda_point_hash_grid_searcher2.h>
#include <jet/vector4.h>

#include <cuda_runtime.h>

namespace jet {

class CudaParticleSystemData2 {
 public:
    //! Scalar int data chunk.
    typedef CudaArray1<int> IntData;

    //! Scalar float data chunk.
    typedef CudaArray1<float> FloatData;

    //! Vector data chunk.
    typedef CudaArray1<float2> VectorData;

    //! Default constructor.
    CudaParticleSystemData2();

    //! Constructs particle system data with given number of particles.
    explicit CudaParticleSystemData2(size_t numberOfParticles);

    //! Copy constructor.
    CudaParticleSystemData2(const CudaParticleSystemData2& other);

    //! Destructor.
    virtual ~CudaParticleSystemData2();

    //!
    //! \brief Resizes the number of particles of the container.
    //!
    //! \param[in]  newNumberOfParticles    New number of particles.
    //!
    void resize(size_t newNumberOfParticles);

    //! Returns the number of particles.
    size_t numberOfParticles() const;

    //!
    size_t addIntData(int initialVal = 0);

    //!
    size_t addFloatData(float initialVal = 0.0f);

    //!
    size_t addVectorData(const Vector2F& initialVal = Vector2F{});

    //!
    size_t numberOfIntData() const;

    //!
    size_t numberOfFloatData() const;

    //!
    size_t numberOfVectorData() const;

    //!
    CudaArrayView1<float2> positions();

    //!
    ConstCudaArrayView1<float2> positions() const;

    //!
    CudaArrayView1<float2> velocities();

    //!
    ConstCudaArrayView1<float2> velocities() const;

    //!
    CudaArrayView1<int> intDataAt(size_t idx);

    //!
    ConstCudaArrayView1<int> intDataAt(size_t idx) const;

    //!
    CudaArrayView1<float> floatDataAt(size_t idx);

    //!
    ConstCudaArrayView1<float> floatDataAt(size_t idx) const;

    //!
    CudaArrayView1<float2> vectorDataAt(size_t idx);

    //!
    ConstCudaArrayView1<float2> vectorDataAt(size_t idx) const;

    //!
    void addParticle(const Vector2F& newPosition,
                     const Vector2F& newVelocity = Vector2F{});

    //!
    void addParticles(
        ConstArrayView1<Vector2F> newPositions,
        ConstArrayView1<Vector2F> newVelocities = ArrayView1<Vector2F>{});

    //!
    void addParticles(
        ConstCudaArrayView1<float2> newPositions,
        ConstCudaArrayView1<float2> newVelocities = CudaArrayView1<float2>{});

    //!
    ConstCudaArrayView1<uint32_t> neighborStarts() const;

    //!
    ConstCudaArrayView1<uint32_t> neighborEnds() const;

    //!
    ConstCudaArrayView1<uint32_t> neighborLists() const;

    const CudaPointHashGridSearcher2* neighborSearcher() const;

    //! Builds neighbor searcher with given search radius.
    void buildNeighborSearcher(float maxSearchRadius);

    //! Builds neighbor lists with given search radius.
    void buildNeighborLists(float maxSearchRadius);

    //! Copies from other particle system data.
    void set(const CudaParticleSystemData2& other);

    //! Copies from other particle system data.
    CudaParticleSystemData2& operator=(const CudaParticleSystemData2& other);

 protected:
    CudaPointHashGridSearcher2Ptr _neighborSearcher;
    CudaArray1<uint32_t> _neighborStarts;
    CudaArray1<uint32_t> _neighborEnds;
    CudaArray1<uint32_t> _neighborLists;

 private:
    size_t _numberOfParticles = 0;
    size_t _positionIdx;
    size_t _velocityIdx;

    std::vector<IntData> _intDataList;
    std::vector<FloatData> _floatDataList;
    std::vector<VectorData> _vectorDataList;
};

//! Shared pointer type of CudaParticleSystemData2.
typedef std::shared_ptr<CudaParticleSystemData2> CudaParticleSystemData2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_PARTICLE_SYSTEM_DATA2_H_

#endif  // JET_USE_CUDA
