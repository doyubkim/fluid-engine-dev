// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_COLLOCATED_VECTOR_GRID_H_
#define INCLUDE_JET_COLLOCATED_VECTOR_GRID_H_

#include <jet/array.h>
#include <jet/array_samplers.h>
#include <jet/vector_grid.h>

namespace jet {

//! \brief Abstract base class for N-D collocated vector grid structure.
template <size_t N>
class CollocatedVectorGrid : public VectorGrid<N> {
 public:
    using typename VectorGrid<N>::VectorDataView;
    using typename VectorGrid<N>::ConstVectorDataView;
    using typename VectorGrid<N>::DataPositionFunc;
    using VectorGrid<N>::gridSpacing;

    //! Constructs an empty grid.
    CollocatedVectorGrid();

    //! Default destructor.
    virtual ~CollocatedVectorGrid();

    //! Returns the actual data point size.
    virtual Vector<size_t, N> dataSize() const = 0;

    //!
    //! \brief Returns data position for the grid point at (0, 0, ...).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    virtual Vector<double, N> dataOrigin() const = 0;

    //! Returns the grid data at given data point.
    const Vector<double, N>& operator()(const Vector<size_t, N>& idx) const;

    //! Returns the grid data at given data point.
    Vector<double, N>& operator()(const Vector<size_t, N>& idx);

    //! Returns the grid data at given data point.
    template <typename... Indices>
    const Vector<double, N>& operator()(size_t i, Indices... indices) const {
        return (*this)(Vector<size_t, N>(i, indices...));
    }

    //! Returns the grid data at given data point.
    template <typename... Indices>
    Vector<double, N>& operator()(size_t i, Indices... indices) {
        return (*this)(Vector<size_t, N>(i, indices...));
    }

    //! Returns divergence at data point location.
    double divergenceAtDataPoint(const Vector<size_t, N>& idx) const;

    //! Returns divergence at data point location.
    template <typename... Indices>
    double divergenceAtDataPoint(size_t i, Indices... indices) const {
        return divergenceAtDataPoint(Vector<size_t, N>(i, indices...));
    }

    //! Returns curl at data point location.
    typename GetCurl<N>::type curlAtDataPoint(
        const Vector<size_t, N>& idx) const;

    //! Returns curl at data point location.
    template <typename... Indices>
    typename GetCurl<N>::type curlAtDataPoint(size_t i,
                                              Indices... indices) const {
        return curlAtDataPoint(Vector<size_t, N>(i, indices...));
    }

    //! Returns the read-write data array view.
    VectorDataView dataView();

    //! Returns the read-only data array view.
    ConstVectorDataView dataView() const;

    //! Returns the function that maps data point to its position.
    DataPositionFunc dataPosition() const;

    //!
    //! \brief Invokes the given function \p func for each data point.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in serial manner. The input parameters are i, j (and k for 3-D)
    //! indices of a data point. The order of execution is i-first, j-next.
    //!
    void forEachDataPointIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const;

    template <size_t M = N>
    std::enable_if_t<M == 2, void> forEachDataPointIndex(
        const std::function<void(size_t, size_t)>& func) const {
        forEachDataPointIndex(
            [&func](const Vector2UZ& idx) { func(idx.x, idx.y); });
    }

    template <size_t M = N>
    std::enable_if_t<M == 3, void> forEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const {
        forEachDataPointIndex(
            [&func](const Vector3UZ& idx) { func(idx.x, idx.y, idx.z); });
    }

    //!
    //! \brief Invokes the given function \p func for each data point
    //! parallelly.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in parallel manner. The input parameters are i, j (and k for 3-D)
    //! indices of a data point. The order of execution can be arbitrary since
    //! it's multi-threaded.
    //!
    void parallelForEachDataPointIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const;

    template <size_t M = N>
    std::enable_if_t<M == 2, void> parallelForEachDataPointIndex(
        const std::function<void(size_t, size_t)>& func) const {
        parallelForEachDataPointIndex(
            [&func](const Vector2UZ& idx) { func(idx.x, idx.y); });
    }

    template <size_t M = N>
    std::enable_if_t<M == 3, void> parallelForEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const {
        parallelForEachDataPointIndex(
            [&func](const Vector3UZ& idx) { func(idx.x, idx.y, idx.z); });
    }

    // VectorField implementations

    //! Returns sampled value at given position \p x.
    Vector<double, N> sample(const Vector<double, N>& x) const override;

    //! Returns divergence at given position \p x.
    double divergence(const Vector<double, N>& x) const override;

    //! Returns curl at given position \p x.
    typename GetCurl<N>::type curl(const Vector<double, N>& x) const override;

    //!
    //! \brief Returns the sampler function.
    //!
    //! This function returns the data sampler function object. The sampling
    //! function is linear.
    //!
    std::function<Vector<double, N>(const Vector<double, N>&)> sampler()
        const override;

 protected:
    using VectorGrid<N>::swapGrid;
    using VectorGrid<N>::setGrid;

    //! Swaps the data storage and predefined samplers with given grid.
    void swapCollocatedVectorGrid(CollocatedVectorGrid* other);

    //! Sets the data storage and predefined samplers with given grid.
    void setCollocatedVectorGrid(const CollocatedVectorGrid& other);

    //! Fetches the data into a continuous linear array.
    void getData(Array1<double>& data) const override;

    //! Sets the data from a continuous linear array.
    void setData(const ConstArrayView1<double>& data) override;

 private:
    Array<Vector<double, N>, N> _data;
    LinearArraySampler<Vector<double, N>, N> _linearSampler;
    std::function<Vector<double, N>(const Vector<double, N>&)> _sampler;

    void onResize(const Vector<size_t, N>& resolution,
                  const Vector<double, N>& gridSpacing,
                  const Vector<double, N>& origin,
                  const Vector<double, N>& initialValue) final;

    void resetSampler();
};

//! 2-D CollocatedVectorGrid type.
using CollocatedVectorGrid2 = CollocatedVectorGrid<2>;

//! 3-D CollocatedVectorGrid type.
using CollocatedVectorGrid3 = CollocatedVectorGrid<3>;

//! Shared pointer for the CollocatedVectorGrid2 type.
using CollocatedVectorGrid2Ptr = std::shared_ptr<CollocatedVectorGrid2>;

//! Shared pointer for the CollocatedVectorGrid3 type.
using CollocatedVectorGrid3Ptr = std::shared_ptr<CollocatedVectorGrid3>;

}  // namespace jet

#endif  // INCLUDE_JET_COLLOCATED_VECTOR_GRID_H_
