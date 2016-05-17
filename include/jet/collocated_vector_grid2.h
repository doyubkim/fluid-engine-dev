// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_COLLOCATED_VECTOR_GRID2_H_
#define INCLUDE_JET_COLLOCATED_VECTOR_GRID2_H_

#include <jet/array2.h>
#include <jet/array_samplers2.h>
#include <jet/vector_grid2.h>

namespace jet {

class CollocatedVectorGrid2 : public VectorGrid2 {
 public:
    CollocatedVectorGrid2();

    virtual ~CollocatedVectorGrid2();

    virtual Size2 dataSize() const = 0;

    //! Returns data position for the grid point at (0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    virtual Vector2D dataOrigin() const = 0;

    const Vector2D& operator()(size_t i, size_t j) const;

    Vector2D& operator()(size_t i, size_t j);

    //! Returns divergence at data point location.
    //! \param i Data index i.
    //! \param j Data index j.
    double divergenceAtDataPoint(size_t i, size_t j) const;

    //! Returns curl at data point location.
    //! \param i Data index i.
    //! \param j Data index j.
    double curlAtDataPoint(size_t i, size_t j) const;

    VectorDataAccessor dataAccessor();

    ConstVectorDataAccessor constDataAccessor() const;

    DataPositionFunc dataPosition() const;

    void forEachDataPointIndex(
        const std::function<void(size_t, size_t)>& func) const;

    void parallelForEachDataPointIndex(
        const std::function<void(size_t, size_t)>& func) const;

    //! Serializes the grid instance to the output stream \p strm.
    void serialize(std::ostream* strm) const override;

    //! Deserializes the input stream \p strm to the grid instance.
    void deserialize(std::istream* strm) override;

    // VectorField2 implementations
    Vector2D sample(const Vector2D& x) const override;

    double divergence(const Vector2D& x) const override;

    double curl(const Vector2D& x) const override;

    std::function<Vector2D(const Vector2D&)> sampler() const override;

 protected:
    void swapCollocatedVectorGrid(CollocatedVectorGrid2* other);

    void setCollocatedVectorGrid(const CollocatedVectorGrid2& other);

 private:
    Array2<Vector2D> _data;
    LinearArraySampler2<Vector2D, double> _linearSampler;
    std::function<Vector2D(const Vector2D&)> _sampler;

    void onResize(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& origin,
        const Vector2D& initialValue) override;

    void resetSampler();
};

}  // namespace jet

#endif  // INCLUDE_JET_COLLOCATED_VECTOR_GRID2_H_
