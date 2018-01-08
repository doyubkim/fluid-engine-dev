// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_PARTICLE_SYSTEM_DATA3_H_
#define INCLUDE_JET_CUDA_PARTICLE_SYSTEM_DATA3_H_

#include <jet/array_view1.h>
#include <jet/cuda_array1.h>
#include <jet/vector4.h>

#include <cuda_runtime.h>

namespace jet {

namespace experimental {

class CudaParticleSystemData3 {
 public:
    //! Scalar int data chunk.
    typedef CudaArray1<int> IntData;

    //! Scalar float data chunk.
    typedef CudaArray1<float> FloatData;

    //! Vector data chunk.
    typedef CudaArray1<float4> VectorData;

    //! Default constructor.
    CudaParticleSystemData3();

    //! Constructs particle system data with given number of particles.
    explicit CudaParticleSystemData3(size_t numberOfParticles);

    //! Copy constructor.
    CudaParticleSystemData3(const CudaParticleSystemData3& other);

    //! Destructor.
    virtual ~CudaParticleSystemData3();

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
    size_t addVectorData(const Vector4F& initialVal = Vector4F{});

    //!
    size_t numberOfIntData() const;

    //!
    size_t numberOfFloatData() const;

    //!
    size_t numberOfVectorData() const;

    //!
    CudaArrayView1<float4> positions();

    //!
    const CudaArrayView1<float4> positions() const;

    //!
    CudaArrayView1<float4> velocities();

    //!
    const CudaArrayView1<float4> velocities() const;

    //!
    CudaArrayView1<int> intDataAt(size_t idx);

    //!
    const CudaArrayView1<int> intDataAt(size_t idx) const;

    //!
    CudaArrayView1<float> floatDataAt(size_t idx);

    //!
    const CudaArrayView1<float> floatDataAt(size_t idx) const;

    //!
    CudaArrayView1<float4> vectorDataAt(size_t idx);

    //!
    const CudaArrayView1<float4> vectorDataAt(size_t idx) const;

    //!
    void addParticle(const Vector4F& newPosition,
                     const Vector4F& newVelocity = Vector4F{});

    //!
    void addParticles(
        const ArrayView1<Vector4F>& newPositions,
        const ArrayView1<Vector4F>& newVelocities = ArrayView1<Vector4F>{});

    //!
    void addParticles(
        const CudaArrayView1<float4>& newPositions,
        const CudaArrayView1<float4>& newVelocities = CudaArrayView1<float4>{});

    //! Copies from other particle system data.
    void set(const CudaParticleSystemData3& other);

    //! Copies from other particle system data.
    CudaParticleSystemData3& operator=(const CudaParticleSystemData3& other);

 private:
    size_t _numberOfParticles = 0;
    size_t _positionIdx;
    size_t _velocityIdx;

    std::vector<IntData> _intDataList;
    std::vector<FloatData> _floatDataList;
    std::vector<VectorData> _vectorDataList;
};

//! Shared pointer type of CudaParticleSystemData3.
typedef std::shared_ptr<CudaParticleSystemData3> CudaParticleSystemData3Ptr;

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_PARTICLE_SYSTEM_DATA3_H_

#endif  // JET_USE_CUDA
