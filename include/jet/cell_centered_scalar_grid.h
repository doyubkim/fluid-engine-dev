// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CELL_CENTERED_SCALAR_GRID_H_
#define INCLUDE_JET_CELL_CENTERED_SCALAR_GRID_H_

#include <jet/scalar_grid.h>

namespace jet {

//!
//! \brief N-D Cell-centered scalar grid structure.
//!
//! This class represents N-D cell-centered scalar grid which extends
//! ScalarGrid. As its name suggests, the class defines the data point at the
//! center of a grid cell. Thus, the dimension of data points are equal to the
//! dimension of the cells.
//!
template <size_t N>
class CellCenteredScalarGrid final : public ScalarGrid<N> {
 public:
    JET_GRID_TYPE_NAME(CellCenteredScalarGrid, N)

    class Builder;

    using ScalarGrid<N>::resize;
    using ScalarGrid<N>::resolution;
    using ScalarGrid<N>::origin;
    using ScalarGrid<N>::gridSpacing;

    //! Constructs zero-sized grid.
    CellCenteredScalarGrid();

    //! Constructs a grid with given resolution, grid spacing, origin and
    //! initial value.
    CellCenteredScalarGrid(
        const Vector<size_t, N>& resolution,
        const Vector<double, N>& gridSpacing =
            Vector<double, N>::makeConstant(1.0),
        const Vector<double, N>& origin = Vector<double, N>(),
        double initialValue = 0.0);

    //! Copy constructor.
    CellCenteredScalarGrid(const CellCenteredScalarGrid& other);

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
    void set(const CellCenteredScalarGrid& other);

    //! Sets the contents with the given \p other grid.
    CellCenteredScalarGrid& operator=(const CellCenteredScalarGrid& other);

    //! Returns the copy of the grid instance.
    std::shared_ptr<ScalarGrid<N>> clone() const override;

    //! Returns builder fox CellCenteredScalarGrid.
    static Builder builder();

protected:
    using ScalarGrid<N>::swapScalarGrid;
    using ScalarGrid<N>::setScalarGrid;
};

//! 2-D CellCenteredScalarGrid type.
using CellCenteredScalarGrid2 = CellCenteredScalarGrid<2>;

//! 3-D CellCenteredScalarGrid type.
using CellCenteredScalarGrid3 = CellCenteredScalarGrid<3>;

//! Shared pointer for the CellCenteredScalarGrid2 type.
using CellCenteredScalarGrid2Ptr = std::shared_ptr<CellCenteredScalarGrid2>;

//! Shared pointer for the CellCenteredScalarGrid3 type.
using CellCenteredScalarGrid3Ptr = std::shared_ptr<CellCenteredScalarGrid3>;

//!
//! \brief Front-end to create CellCenteredScalarGrid objects step by step.
//!
template <size_t N>
class CellCenteredScalarGrid<N>::Builder final : public ScalarGridBuilder<N> {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Vector<size_t, N>& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(const Vector<double, N>& gridSpacing);

    //! Returns builder with grid origin.
    Builder& withOrigin(const Vector<double, N>& gridOrigin);

    //! Returns builder with initial value.
    Builder& withInitialValue(double initialVal);

    //! Builds CellCenteredScalarGrid instance.
    CellCenteredScalarGrid<N> build() const;

    //! Builds shared pointer of CellCenteredScalarGrid instance.
    std::shared_ptr<CellCenteredScalarGrid<N>> makeShared() const;

    //!
    //! \brief Builds shared pointer of CellCenteredScalarGrid instance.
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
    Vector<size_t, N> _resolution = (Vector<size_t, N>::makeConstant(1.0));
    Vector<double, N> _gridSpacing = (Vector<double, N>::makeConstant(1.0));
    Vector<double, N> _gridOrigin = (Vector<double, N>());
    double _initialVal = 0.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_CELL_CENTERED_SCALAR_GRID_H_
