// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SCALAR_GRID_H_
#define INCLUDE_JET_SCALAR_GRID_H_

#include <jet/array.h>
#include <jet/array_samplers.h>
#include <jet/array_view.h>
#include <jet/grid.h>
#include <jet/parallel.h>
#include <jet/scalar_field.h>

namespace jet {

//! Abstract base class for N-D scalar grid structure.
template <size_t N>
class ScalarGrid : public ScalarField<N>, public Grid<N> {
 public:
    //! Read-write array view type.
    using ScalarDataView = ArrayView<double, N>;

    //! Read-only array view type.
    using ConstScalarDataView = ArrayView<const double, N>;

    // Import Grid members
    using Grid<N>::gridSpacing;
    using Grid<N>::origin;
    using Grid<N>::resolution;

    //! Constructs an empty grid.
    ScalarGrid();

    //! Default destructor.
    virtual ~ScalarGrid();

    //!
    //! \brief Returns the size of the grid data.
    //!
    //! This function returns the size of the grid data which is not necessarily
    //! equal to the grid resolution if the data is not stored at cell-center.
    //!
    virtual Vector<size_t, N> dataSize() const = 0;

    //!
    //! \brief Returns the origin of the grid data.
    //!
    //! This function returns data position for the grid point at (0, 0, ...).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    virtual Vector<double, N> dataOrigin() const = 0;

    //! Returns the copy of the grid instance.
    virtual std::shared_ptr<ScalarGrid> clone() const = 0;

    //! Clears the contents of the grid.
    void clear();

    //! Resizes the grid using given parameters.
    void resize(const Vector<size_t, N>& resolution,
                const Vector<double, N>& gridSpacing =
                    Vector<double, N>::makeConstant(1),
                const Vector<double, N>& origin = Vector<double, N>(),
                double initialValue = 0.0);

    //! Resizes the grid using given parameters.
    void resize(const Vector<double, N>& gridSpacing,
                const Vector<double, N>& origin);

    //! Returns the grid data at given data point.
    const double& operator()(const Vector<size_t, N>& idx) const;

    //! Returns the grid data at given data point.
    double& operator()(const Vector<size_t, N>& idx);

    //! Returns the grid data at given data point.
    template <typename... Indices>
    const double& operator()(size_t i, Indices... indices) const {
        return (*this)(Vector<size_t, N>(i, indices...));
    }

    //! Returns the grid data at given data point.
    template <typename... Indices>
    double& operator()(size_t i, Indices... indices) {
        return (*this)(Vector<size_t, N>(i, indices...));
    }

    //! Returns the gradient vector at given data point.
    Vector<double, N> gradientAtDataPoint(const Vector<size_t, N>& idx) const;

    //! Returns the gradient vector at given data point.
    template <typename... Indices>
    Vector<double, N> gradientAtDataPoint(size_t i, Indices... indices) const {
        return gradientAtDataPoint(Vector<size_t, N>(i, indices...));
    }

    //! Returns the Laplacian at given data point.
    double laplacianAtDataPoint(const Vector<size_t, N>& idx) const;

    //! Returns the Laplacian at given data point.
    template <typename... Indices>
    double laplacianAtDataPoint(size_t i, Indices... indices) const {
        return laplacianAtDataPoint(Vector<size_t, N>(i, indices...));
    }

    //! Returns the read-write data array accessor.
    ScalarDataView dataView();

    //! Returns the read-only data array accessor.
    ConstScalarDataView dataView() const;

    //! Returns the function that maps data point to its position.
    GridDataPositionFunc<N> dataPosition() const;

    //! Fills the grid with given value.
    void fill(double value,
              ExecutionPolicy policy = ExecutionPolicy::kParallel);

    //! Fills the grid with given position-to-value mapping function.
    void fill(const std::function<double(const Vector<double, N>&)>& func,
              ExecutionPolicy policy = ExecutionPolicy::kParallel);

    //!
    //! \brief Invokes the given function \p func for each data point.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in serial manner. The input parameters are i, j, ... indices of a
    //! data point. The order of execution is i-first, j-next.
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
    //! in parallel.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in parallel manner. The input parameters are i, j, ... indices of
    //! a data point. The order of execution can be arbitrary since it's
    //! multi-threaded.
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

    // ScalarField implementations

    //!
    //! \brief Returns the sampled value at given position \p x.
    //!
    //! This function returns the data sampled at arbitrary position \p x.
    //! The sampling function is linear.
    //!
    double sample(const Vector<double, N>& x) const override;

    //!
    //! \brief Returns the sampler function.
    //!
    //! This function returns the data sampler function object. The sampling
    //! function is linear.
    //!
    std::function<double(const Vector<double, N>&)> sampler() const override;

    //! Returns the gradient vector at given position \p x.
    Vector<double, N> gradient(const Vector<double, N>& x) const override;

    //! Returns the Laplacian at given position \p x.
    double laplacian(const Vector<double, N>& x) const override;

    //! Serializes the grid instance to the output buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the input buffer to the grid instance.
    void deserialize(const std::vector<uint8_t>& buffer) override;

 protected:
    using Grid<N>::setSizeParameters;
    using Grid<N>::swapGrid;
    using Grid<N>::setGrid;

    //! Swaps the data storage and predefined samplers with given grid.
    void swapScalarGrid(ScalarGrid* other);

    //! Sets the data storage and predefined samplers with given grid.
    void setScalarGrid(const ScalarGrid& other);

    //! Fetches the data into a continuous linear array.
    void getData(Array1<double>& data) const override;

    //! Sets the data from a continuous linear array.
    void setData(const ConstArrayView1<double>& data) override;

 private:
    Array<double, N> _data;
    LinearArraySampler<double, N> _linearSampler;
    std::function<double(const Vector<double, N>&)> _sampler;

    void resetSampler();
};

//! 2-D ScalarGrid type.
using ScalarGrid2 = ScalarGrid<2>;

//! 3-D ScalarGrid type.
using ScalarGrid3 = ScalarGrid<3>;

//! Shared pointer for the ScalarGrid2 type.
using ScalarGrid2Ptr = std::shared_ptr<ScalarGrid2>;

//! Shared pointer for the ScalarGrid3 type.
using ScalarGrid3Ptr = std::shared_ptr<ScalarGrid3>;

//! Abstract base class for N-D scalar grid builder.
template <size_t N>
class ScalarGridBuilder {
 public:
    //! Creates a builder.
    ScalarGridBuilder();

    //! Default destructor.
    virtual ~ScalarGridBuilder();

    //! Returns N-D scalar grid with given parameters.
    virtual std::shared_ptr<ScalarGrid<N>> build(
        const Vector<size_t, N>& resolution,
        const Vector<double, N>& gridSpacing,
        const Vector<double, N>& gridOrigin, double initialVal) const = 0;
};

//! 2-D ScalarGridBuilder type.
using ScalarGridBuilder2 = ScalarGridBuilder<2>;

//! 3-D ScalarGridBuilder type.
using ScalarGridBuilder3 = ScalarGridBuilder<3>;

//! Shared pointer for the ScalarGridBuilder2 type.
using ScalarGridBuilder2Ptr = std::shared_ptr<ScalarGridBuilder2>;

//! Shared pointer for the ScalarGridBuilder3 type.
using ScalarGridBuilder3Ptr = std::shared_ptr<ScalarGridBuilder3>;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_GRID_H_
