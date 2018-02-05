// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_COLLOCATED_VECTOR_GRID3_H_
#define INCLUDE_JET_COLLOCATED_VECTOR_GRID3_H_

#include <jet/array3.h>
#include <jet/array_samplers3.h>
#include <jet/vector_grid3.h>
#include <vector>

namespace jet {

//! \brief Abstract base class for 3-D collocated vector grid structure.
class CollocatedVectorGrid3 : public VectorGrid3 {
 public:
    //! Constructs an empty grid.
    CollocatedVectorGrid3();

    //! Default destructor.
    virtual ~CollocatedVectorGrid3();

    //! Returns the actual data point size.
    virtual Size3 dataSize() const = 0;

    //!
    //! \brief Returns data position for the grid point at (0, 0, 0).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    virtual Vector3D dataOrigin() const = 0;

    //! Returns the grid data at given data point.
    const Vector3D& operator()(size_t i, size_t j, size_t k) const;

    //! Returns the grid data at given data point.
    Vector3D& operator()(size_t i, size_t j, size_t k);

    //! Returns divergence at data point location.
    double divergenceAtDataPoint(size_t i, size_t j, size_t k) const;

    //! Returns curl at data point location.
    Vector3D curlAtDataPoint(size_t i, size_t j, size_t k) const;

    //! Returns the read-write data array accessor.
    VectorDataAccessor dataAccessor();

    //! Returns the read-only data array accessor.
    ConstVectorDataAccessor constDataAccessor() const;

    //! Returns the function that maps data point to its position.
    DataPositionFunc dataPosition() const;

    //!
    //! \brief Invokes the given function \p func for each data point.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in serial manner. The input parameters are i and j indices of a
    //! data point. The order of execution is i-first, j-last.
    //!
    void forEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each data point
    //! parallelly.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in parallel manner. The input parameters are i and j indices of a
    //! data point. The order of execution can be arbitrary since it's
    //! multi-threaded.
    //!
    void parallelForEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    // VectorField3 implementations

    //! Returns sampled value at given position \p x.
    Vector3D sample(const Vector3D& x) const override;

    //! Returns divergence at given position \p x.
    double divergence(const Vector3D& x) const override;

    //! Returns curl at given position \p x.
    Vector3D curl(const Vector3D& x) const override;

    //!
    //! \brief Returns the sampler function.
    //!
    //! This function returns the data sampler function object. The sampling
    //! function is linear.
    //!
    std::function<Vector3D(const Vector3D&)> sampler() const override;

 protected:
    //! Swaps the data storage and predefined samplers with given grid.
    void swapCollocatedVectorGrid(CollocatedVectorGrid3* other);

    //! Sets the data storage and predefined samplers with given grid.
    void setCollocatedVectorGrid(const CollocatedVectorGrid3& other);

    //! Fetches the data into a continuous linear array.
    void getData(std::vector<double>* data) const override;

    //! Sets the data from a continuous linear array.
    void setData(const std::vector<double>& data) override;

 private:
    Array3<Vector3D> _data;
    LinearArraySampler3<Vector3D, double> _linearSampler;
    std::function<Vector3D(const Vector3D&)> _sampler;

    void onResize(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& origin,
        const Vector3D& initialValue) final;

    void resetSampler();
};

//! Shared pointer for the CollocatedVectorGrid3 type.
typedef std::shared_ptr<CollocatedVectorGrid3> CollocatedVectorGrid3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_COLLOCATED_VECTOR_GRID3_H_
