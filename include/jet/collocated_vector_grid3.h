// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_COLLOCATED_VECTOR_GRID3_H_
#define INCLUDE_JET_COLLOCATED_VECTOR_GRID3_H_

#include <jet/array3.h>
#include <jet/array_samplers3.h>
#include <jet/vector_grid3.h>

namespace jet {

class CollocatedVectorGrid3 : public VectorGrid3 {
 public:
    CollocatedVectorGrid3();

    virtual ~CollocatedVectorGrid3();

    virtual Size3 dataSize() const = 0;

    //! Returns data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    virtual Vector3D dataOrigin() const = 0;

    const Vector3D& operator()(size_t i, size_t j, size_t k) const;

    Vector3D& operator()(size_t i, size_t j, size_t k);

    //! Returns divergence at data point location.
    //! \param i Data index i.
    //! \param j Data index j.
    //! \param k Data index k.
    double divergenceAtDataPoint(size_t i, size_t j, size_t k) const;

    //! Returns curl at data point location.
    //! \param i Data index i.
    //! \param j Data index j.
    //! \param k Data index k.
    Vector3D curlAtDataPoint(size_t i, size_t j, size_t k) const;

    VectorDataAccessor dataAccessor();

    ConstVectorDataAccessor constDataAccessor() const;

    DataPositionFunc dataPosition() const;

    void forEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    void parallelForEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //! Serializes the grid instance to the output stream \p strm.
    void serialize(std::ostream* strm) const override;

    //! Deserializes the input stream \p strm to the grid instance.
    void deserialize(std::istream* strm) override;

    // VectorField3 implementations
    Vector3D sample(const Vector3D& x) const override;

    double divergence(const Vector3D& x) const override;

    Vector3D curl(const Vector3D& x) const override;

    std::function<Vector3D(const Vector3D&)> sampler() const override;

 protected:
    void swapCollocatedVectorGrid(CollocatedVectorGrid3* other);

    void setCollocatedVectorGrid(const CollocatedVectorGrid3& other);

 private:
    Array3<Vector3D> _data;
    LinearArraySampler3<Vector3D, double> _linearSampler;
    std::function<Vector3D(const Vector3D&)> _sampler;

    void onResize(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& origin,
        const Vector3D& initialValue) override;

    void resetSampler();
};

}  // namespace jet

#endif  // INCLUDE_JET_COLLOCATED_VECTOR_GRID3_H_
