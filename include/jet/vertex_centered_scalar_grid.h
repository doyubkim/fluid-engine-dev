// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID_H_
#define INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID_H_

#include <jet/array.h>
#include <jet/scalar_grid.h>

namespace jet {

//!
//! \brief N-D Vertex-centered scalar grid structure.
//!
//! This class represents N-D vertex-centered scalar grid which extends
//! ScalarGrid. As its name suggests, the class defines the data point at the
//! grid vertices (corners). Thus, A x B x ... grid resolution will have (A+1) x
//! (B+1) x ... data points.
//!
template <size_t N>
class VertexCenteredScalarGrid final : public ScalarGrid<N> {
 public:
    JET_GRID_TYPE_NAME(VertexCenteredScalarGrid, N)

    class Builder;

    using ScalarGrid<N>::resize;
    using ScalarGrid<N>::resolution;
    using ScalarGrid<N>::origin;

    //! Constructs zero-sized grid.
    VertexCenteredScalarGrid();

    //! Constructs a grid with given resolution, grid spacing, origin and
    //! initial value.
    VertexCenteredScalarGrid(
        const Vector<size_t, N>& resolution,
        const Vector<double, N>& gridSpacing =
            Vector<double, N>::makeConstant(1),
        const Vector<double, N>& origin = Vector<double, N>(),
        double initialValue = 0.0);

    //! Copy constructor.
    VertexCenteredScalarGrid(const VertexCenteredScalarGrid& other);

    //! Returns the actual data point size.
    Vector<size_t, N> dataSize() const override;

    //! Returns data position for the grid point at (0, 0, ...).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector<double, N> dataOrigin() const override;

    //! Returns the copy of the grid instance.
    std::shared_ptr<ScalarGrid<N>> clone() const override;

    //!
    //! \brief Swaps the contents with the given \p other grid.
    //!
    //! This function swaps the contents of the grid instance with the given
    //! grid object \p other only if \p other has the same type with this grid.
    //!
    void swap(Grid<N>* other) override;

    //! Sets the contents with the given \p other grid.
    void set(const VertexCenteredScalarGrid<N>& other);

    //! Sets the contents with the given \p other grid.
    VertexCenteredScalarGrid<N>& operator=(
        const VertexCenteredScalarGrid<N>& other);

    //! Returns builder fox VertexCenteredScalarGrid<N>.
    static Builder builder();

 protected:
    using ScalarGrid<N>::swapScalarGrid;
    using ScalarGrid<N>::setScalarGrid;
};

//! 2-D VertexCenteredScalarGrid type.
using VertexCenteredScalarGrid2 = VertexCenteredScalarGrid<2>;

//! 3-D VertexCenteredScalarGrid type.
using VertexCenteredScalarGrid3 = VertexCenteredScalarGrid<3>;

//! Shared pointer for the VertexCenteredScalarGrid2 type.
using VertexCenteredScalarGrid2Ptr = std::shared_ptr<VertexCenteredScalarGrid2>;

//! Shared pointer for the VertexCenteredScalarGrid3 type.
using VertexCenteredScalarGrid3Ptr = std::shared_ptr<VertexCenteredScalarGrid3>;

//!
//! \brief Front-end to create VertexCenteredScalarGrid objects step by step.
//!
template <size_t N>
class VertexCenteredScalarGrid<N>::Builder final : public ScalarGridBuilder<N> {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Vector<size_t, N>& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(const Vector<double, N>& gridSpacing);

    //! Returns builder with grid origin.
    Builder& withOrigin(const Vector<double, N>& gridOrigin);

    //! Returns builder with initial value.
    Builder& withInitialValue(double initialVal);

    //! Builds VertexCenteredScalarGrid<N> instance.
    VertexCenteredScalarGrid<N> build() const;

    //! Builds shared pointer of VertexCenteredScalarGrid<N> instance.
    std::shared_ptr<VertexCenteredScalarGrid<N>> makeShared() const;

    //!
    //! \brief Builds shared pointer of VertexCenteredScalarGrid<N> instance.
    //!
    //! This is an overriding function that implements ScalarGridBuilder2.
    //!
    std::shared_ptr<ScalarGrid<N>> build(const Vector<size_t, N>& resolution,
                                         const Vector<double, N>& gridSpacing,
                                         const Vector<double, N>& gridOrigin,
                                         double initialVal) const override;

 private:
    // parentheses around some of the initialization expressions due to:
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=52595
    Vector<size_t, N> _resolution = (Vector<size_t, N>::makeConstant(1));
    Vector<double, N> _gridSpacing = (Vector<double, N>::makeConstant(1.0));
    Vector<double, N> _gridOrigin;
    double _initialVal = 0.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID_H_
