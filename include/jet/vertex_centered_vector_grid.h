// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID_H_
#define INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID_H_

#include <jet/collocated_vector_grid.h>

namespace jet {

//!
//! \brief N-D Vertex-centered vector grid structure.
//!
//! This class represents N-D vertex-centered vector grid which extends
//! CollocatedVectorGrid. As its name suggests, the class defines the data
//! point at the grid vertices (corners). Thus, A x B x ... grid resolution will
//! have (A+1) x (B+1) x ... data points.
//!
template <size_t N>
class VertexCenteredVectorGrid final : public CollocatedVectorGrid<N> {
 public:
    JET_GRID_TYPE_NAME(VertexCenteredVectorGrid, N)

    class Builder;

    using CollocatedVectorGrid<N>::resize;
    using CollocatedVectorGrid<N>::resolution;
    using CollocatedVectorGrid<N>::origin;
    using CollocatedVectorGrid<N>::gridSpacing;
    using CollocatedVectorGrid<N>::dataView;
    using CollocatedVectorGrid<N>::dataPosition;

    //! Constructs zero-sized grid.
    VertexCenteredVectorGrid();

    //! Constructs a grid with given resolution, grid spacing, origin and
    //! initial value.
    VertexCenteredVectorGrid(
        const Vector<size_t, N>& resolution,
        const Vector<double, N>& gridSpacing =
            Vector<double, N>::makeConstant(1),
        const Vector<double, N>& origin = Vector<double, N>(),
        const Vector<double, N>& initialValue = Vector<double, N>());

    //! Copy constructor.
    VertexCenteredVectorGrid(const VertexCenteredVectorGrid& other);

    //! Returns the actual data point size.
    Vector<size_t, N> dataSize() const override;

    //! Returns data position for the grid point at (0, 0, ...).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector<double, N> dataOrigin() const override;

    //!
    //! \brief Swaps the contents with the given \p other grid.
    //!
    //! This function swaps the contents of the grid instance with the given
    //! grid object \p other only if \p other has the same type with this grid.
    //!
    void swap(Grid<N>* other) override;

    //! Sets the contents with the given \p other grid.
    void set(const VertexCenteredVectorGrid& other);

    //! Sets the contents with the given \p other grid.
    VertexCenteredVectorGrid& operator=(const VertexCenteredVectorGrid& other);

    //! Fills the grid with given value.
    void fill(const Vector<double, N>& value,
              ExecutionPolicy policy = ExecutionPolicy::kParallel) override;

    //! Fills the grid with given function.
    void fill(
        const std::function<Vector<double, N>(const Vector<double, N>&)>& func,
        ExecutionPolicy policy = ExecutionPolicy::kParallel) override;

    //! Returns the copy of the grid instance.
    std::shared_ptr<VectorGrid<N>> clone() const override;

    //! Returns builder fox VertexCenteredVectorGrid.
    static Builder builder();

 protected:
    using CollocatedVectorGrid<N>::swapCollocatedVectorGrid;
    using CollocatedVectorGrid<N>::setCollocatedVectorGrid;
};

//! 2-D VertexCenteredVectorGrid type.
using VertexCenteredVectorGrid2 = VertexCenteredVectorGrid<2>;

//! 3-D VertexCenteredVectorGrid type.
using VertexCenteredVectorGrid3 = VertexCenteredVectorGrid<3>;

//! Shared pointer for the VertexCenteredVectorGrid2 type.
using VertexCenteredVectorGrid2Ptr = std::shared_ptr<VertexCenteredVectorGrid2>;

//! Shared pointer for the VertexCenteredVectorGrid3 type.
using VertexCenteredVectorGrid3Ptr = std::shared_ptr<VertexCenteredVectorGrid3>;

//! A grid builder class that returns N-D vertex-centered vector grid.
template <size_t N>
class VertexCenteredVectorGrid<N>::Builder final : public VectorGridBuilder<N> {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Vector<size_t, N>& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(const Vector<double, N>& gridSpacing);

    //! Returns builder with grid origin.
    Builder& withOrigin(const Vector<double, N>& gridOrigin);

    //! Returns builder with initial value.
    Builder& withInitialValue(const Vector<double, N>& initialVal);

    //! Builds VertexCenteredVectorGrid instance.
    VertexCenteredVectorGrid build() const;

    //! Builds shared pointer of VertexCenteredVectorGrid instance.
    std::shared_ptr<VertexCenteredVectorGrid<N>> makeShared() const;

    //!
    //! \brief Builds shared pointer of VertexCenteredVectorGrid instance.
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

#endif  // INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID_H_
