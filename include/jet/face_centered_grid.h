// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FACE_CENTERED_GRID_H_
#define INCLUDE_JET_FACE_CENTERED_GRID_H_

#include <jet/array.h>
#include <jet/array_samplers.h>
#include <jet/parallel.h>
#include <jet/vector_grid.h>

namespace jet {

//!
//! \brief N-D face-centered (a.k.a MAC or staggered) grid.
//!
//! This class implements face-centered grid which is also known as
//! marker-and-cell (MAC) or staggered grid. This vector grid stores each vector
//! component at face center. Thus, u, v (and w for 3-D) components are not
//! collocated.
//!
template <size_t N>
class FaceCenteredGrid final : public VectorGrid<N> {
 public:
    JET_GRID_TYPE_NAME(FaceCenteredGrid, N)

    class Builder;

    using VectorGrid<N>::resize;
    using VectorGrid<N>::resolution;
    using VectorGrid<N>::gridSpacing;

    using typename VectorGrid<N>::DataPositionFunc;

    //! Read-write scalar data view type.
    typedef ArrayView<double, N> ScalarDataView;

    //! Read-only scalar data view type.
    typedef ArrayView<const double, N> ConstScalarDataView;

    //! Constructs empty grid.
    FaceCenteredGrid();

    //! Resizes the grid using given parameters.
    FaceCenteredGrid(
        const Vector<size_t, N>& resolution,
        const Vector<double, N>& gridSpacing =
            Vector<double, N>::makeConstant(1.0),
        const Vector<double, N>& origin = Vector<double, N>(),
        const Vector<double, N>& initialValue = Vector<double, N>());

    //! Copy constructor.
    FaceCenteredGrid(const FaceCenteredGrid& other);

    //!
    //! \brief Swaps the contents with the given \p other grid.
    //!
    //! This function swaps the contents of the grid instance with the given
    //! grid object \p other only if \p other has the same type with this grid.
    //!
    void swap(Grid<N>* other) override;

    //! Sets the contents with the given \p other grid.
    void set(const FaceCenteredGrid& other);

    //! Sets the contents with the given \p other grid.
    FaceCenteredGrid& operator=(const FaceCenteredGrid& other);

    //! Returns u-value at given data point.
    double& u(const Vector<size_t, N>& idx);

    //! Returns u-value at given data point.
    template <typename... Indices>
    double& u(size_t i, Indices... indices) {
        return u(Vector<size_t, N>(i, indices...));
    }

    //! Returns u-value at given data point.
    const double& u(const Vector<size_t, N>& idx) const;

    //! Returns u-value at given data point.
    template <typename... Indices>
    const double& u(size_t i, Indices... indices) const {
        return u(Vector<size_t, N>(i, indices...));
    }

    //! Returns v-value at given data point.
    double& v(const Vector<size_t, N>& idx);

    //! Returns v-value at given data point.
    template <typename... Indices>
    double& v(size_t i, Indices... indices) {
        return v(Vector<size_t, N>(i, indices...));
    }

    //! Returns v-value at given data point.
    const double& v(const Vector<size_t, N>& idx) const;

    //! Returns v-value at given data point.
    template <typename... Indices>
    const double& v(size_t i, Indices... indices) const {
        return v(Vector<size_t, N>(i, indices...));
    }

    //! Returns w-value at given data point.
    template <size_t M = N>
    std::enable_if_t<M == 3, double&> w(const Vector<size_t, N>& idx) {
        return _data[2](idx);
    }

    //! Returns w-value at given data point.
    template <size_t M = N, typename... Indices>
    std::enable_if_t<M == 3, double&> w(size_t i, Indices... indices) {
        return w(Vector<size_t, N>(i, indices...));
    }

    //! Returns w-value at given data point.
    template <size_t M = N>
    std::enable_if_t<M == 3, const double&> w(
        const Vector<size_t, N>& idx) const {
        return _data[2](idx);
    }

    //! Returns w-value at given data point.
    template <size_t M = N, typename... Indices>
    std::enable_if_t<M == 3, const double&> w(size_t i,
                                              Indices... indices) const {
        return w(Vector<size_t, N>(i, indices...));
    }

    //! Returns interpolated value at cell center.
    Vector<double, N> valueAtCellCenter(const Vector<size_t, N>& idx) const;

    //! Returns interpolated value at cell center.
    template <typename... Indices>
    Vector<double, N> valueAtCellCenter(size_t i, Indices... indices) const {
        return valueAtCellCenter(Vector<size_t, N>(i, indices...));
    }

    //! Returns divergence at cell-center location.
    double divergenceAtCellCenter(const Vector<size_t, N>& idx) const;

    //! Returns divergence at cell-center location.
    template <typename... Indices>
    double divergenceAtCellCenter(size_t i, Indices... indices) const {
        return divergenceAtCellCenter(Vector<size_t, N>(i, indices...));
    }

    //! Returns curl at cell-center location.
    typename GetCurl<N>::type curlAtCellCenter(
        const Vector<size_t, N>& idx) const;

    //! Returns curl at cell-center location.
    template <typename... Indices>
    typename GetCurl<N>::type curlAtCellCenter(size_t i,
                                               Indices... indices) const {
        return curlAtCellCenter(Vector<size_t, N>(i, indices...));
    }

    //! Returns u data view.
    ScalarDataView uView();

    //! Returns read-only u data view.
    ConstScalarDataView uView() const;

    //! Returns v data view.
    ScalarDataView vView();

    //! Returns read-only v data view.
    ConstScalarDataView vView() const;

    //! Returns w data view.
    template <size_t M = N>
    std::enable_if_t<M == 3, ScalarDataView> wView() {
        return dataView(2);
    }

    //! Returns read-only w data view.
    template <size_t M = N>
    std::enable_if_t<M == 3, ConstScalarDataView> wView() const {
        return dataView(2);
    }

    //! Returns i-th data view.
    ScalarDataView dataView(size_t i);

    //! Returns read-only i-th data view.
    ConstScalarDataView dataView(size_t i) const;

    //! Returns function object that maps u data point to its actual position.
    DataPositionFunc uPosition() const;

    //! Returns function object that maps v data point to its actual position.
    DataPositionFunc vPosition() const;

    //! Returns function object that maps w data point to its actual position.
    template <size_t M = N>
    std::enable_if_t<M == 3, DataPositionFunc> wPosition() const {
        return dataPosition(2);
    }

    //! Returns function object that maps data point to its actual position.
    DataPositionFunc dataPosition(size_t i) const;

    //! Returns data size of the u component.
    Vector<size_t, N> uSize() const;

    //! Returns data size of the v component.
    Vector<size_t, N> vSize() const;

    //! Returns data size of the w component.
    template <size_t M = N>
    std::enable_if_t<M == 3, Vector<size_t, N>> wSize() const {
        return dataSize(2);
    }

    //! Returns data size of the i-th component.
    Vector<size_t, N> dataSize(size_t i) const;

    //!
    //! \brief Returns u-data position for the grid point at (0, 0, ...).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    Vector<double, N> uOrigin() const;

    //!
    //! \brief Returns v-data position for the grid point at (0, 0, ...).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    Vector<double, N> vOrigin() const;

    //!
    //! \brief Returns w-data position for the grid point at (0, 0, ...).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    template <size_t M = N>
    std::enable_if_t<M == 3, Vector<double, N>> wOrigin() const {
        return dataOrigin(2);
    }

    //!
    //! \brief Returns i-th data position for the grid point at (0, 0, ...).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    Vector<double, N> dataOrigin(size_t i) const;

    //! Fills the grid with given value.
    void fill(const Vector<double, N>& value,
              ExecutionPolicy policy = ExecutionPolicy::kParallel) override;

    //! Fills the grid with given function.
    void fill(
        const std::function<Vector<double, N>(const Vector<double, N>&)>& func,
        ExecutionPolicy policy = ExecutionPolicy::kParallel) override;

    //! Returns the copy of the grid instance.
    std::shared_ptr<VectorGrid<N>> clone() const override;

    //!
    //! \brief Invokes the given function \p func for each u-data point.
    //!
    //! This function invokes the given function object \p func for each u-data
    //! point in serial manner. The input parameters are i, j (and k for 3-D)
    //! indices of a u-data point. The order of execution is i-first, j-next.
    //!
    void forEachUIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each u-data point in
    //! parallel.
    //!
    //! This function invokes the given function object \p func for each u-data
    //! point in parallel manner. The input parameters are i, j (and k for 3-D)
    //! indices of a u-data point. The order of execution can be arbitrary since
    //! it's multi-threaded.
    //!
    void parallelForEachUIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each v-data point.
    //!
    //! This function invokes the given function object \p func for each v-data
    //! point in serial manner. The input parameters are i, j (and k for 3-D)
    //! indices of a v-data point. The order of execution is i-first, j-next.
    //!
    void forEachVIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each v-data point
    //! parallelly.
    //!
    //! This function invokes the given function object \p func for each v-data
    //! point in parallel manner. The input parameters are i, j (and k for 3-D)
    //! indices of a v-data point. The order of execution can be arbitrary since
    //! it's multi-threaded.
    //!
    void parallelForEachVIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each w-data point.
    //!
    //! This function invokes the given function object \p func for each w-data
    //! point in serial manner. The input parameters are i, j (and k for 3-D)
    //! indices of a w-data point. The order of execution is i-first, j-next.
    //!
    template <size_t M = N>
    std::enable_if_t<M == 3, void> forEachWIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const {
        forEachIndex(dataSize(2), GetUnroll<void, N>::unroll(func));
    }

    //!
    //! \brief Invokes the given function \p func for each w-data point
    //! parallelly.
    //!
    //! This function invokes the given function object \p func for each w-data
    //! point in parallel manner. The input parameters are i, j (and k for 3-D)
    //! indices of a w-data point. The order of execution can be arbitrary since
    //! it's multi-threaded.
    //!
    template <size_t M = N>
    std::enable_if_t<M == 3, void> parallelForEachWIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const {
        parallelForEachIndex(dataSize(2), GetUnroll<void, N>::unroll(func));
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

    //! Returns builder fox FaceCenteredGrid.
    static Builder builder();

 protected:
    using VectorGrid<N>::swapGrid;
    using VectorGrid<N>::setGrid;

    // VectorGrid<N> implementations
    void onResize(const Vector<size_t, N>& resolution,
                  const Vector<double, N>& gridSpacing,
                  const Vector<double, N>& origin,
                  const Vector<double, N>& initialValue) final;

    //! Fetches the data into a continuous linear array.
    void getData(Array1<double>& data) const override;

    //! Sets the data from a continuous linear array.
    void setData(const ConstArrayView1<double>& data) override;

 private:
    std::array<Array<double, N>, N> _data;
    std::array<Vector<double, N>, N> _dataOrigins;
    std::array<LinearArraySampler<double, N>, N> _linearSamplers;
    std::function<Vector<double, N>(const Vector<double, N>&)> _sampler;

    void resetSampler();
};

//! 2-D FaceCenteredGrid type.
using FaceCenteredGrid2 = FaceCenteredGrid<2>;

//! 3-D FaceCenteredGrid type.
using FaceCenteredGrid3 = FaceCenteredGrid<3>;

//! Shared pointer type for the FaceCenteredGrid2.
using FaceCenteredGrid2Ptr = std::shared_ptr<FaceCenteredGrid2>;

//! Shared pointer type for the FaceCenteredGrid3.
using FaceCenteredGrid3Ptr = std::shared_ptr<FaceCenteredGrid3>;

//!
//! \brief Front-end to create FaceCenteredGrid objects step by step.
//!
template <size_t N>
class FaceCenteredGrid<N>::Builder final : public VectorGridBuilder<N> {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Vector<size_t, N>& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(const Vector<double, N>& gridSpacing);

    //! Returns builder with grid origin.
    Builder& withOrigin(const Vector<double, N>& gridOrigin);

    //! Returns builder with initial value.
    Builder& withInitialValue(const Vector<double, N>& initialVal);

    //! Builds FaceCenteredGrid instance.
    FaceCenteredGrid build() const;

    //! Builds shared pointer of FaceCenteredGrid instance.
    std::shared_ptr<FaceCenteredGrid> makeShared() const;

    //!
    //! \brief Builds shared pointer of FaceCenteredGrid instance.
    //!
    //! This is an overriding function that implements VectorGridBuilder2.
    //!
    std::shared_ptr<VectorGrid<N>> build(
        const Vector<size_t, N>& resolution,
        const Vector<double, N>& gridSpacing,
        const Vector<double, N>& gridOrigin,
        const Vector<double, N>& initialVal) const override;

 private:
    Vector<size_t, N> _resolution = Vector<size_t, N>::makeConstant(1);
    Vector<double, N> _gridSpacing = Vector<double, N>::makeConstant(1.0);
    Vector<double, N> _gridOrigin;
    Vector<double, N> _initialVal;
};

}  // namespace jet

#endif  // INCLUDE_JET_FACE_CENTERED_GRID_H_
