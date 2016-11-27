// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID2_H_
#define INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID2_H_

#include <jet/collocated_vector_grid2.h>
#include <utility>  // just make cpplint happy..

namespace jet {

//!
//! \brief 2-D Vertex-centered vector grid structure.
//!
//! This class represents 2-D vertex-centered vector grid which extends
//! CollocatedVectorGrid2. As its name suggests, the class defines the data
//! point at the grid vertices (corners). Thus, A x B grid resolution will have
//! (A+1) x (B+1) data points.
//!
class VertexCenteredVectorGrid2 final : public CollocatedVectorGrid2 {
 public:
    //! Constructs zero-sized grid.
    VertexCenteredVectorGrid2();

    //! Constructs a grid with given resolution, grid spacing, origin and
    //! initial value.
    VertexCenteredVectorGrid2(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double initialValueU = 0.0,
        double initialValueV = 0.0);

    //! Constructs a grid with given resolution, grid spacing, origin and
    //! initial value.
    VertexCenteredVectorGrid2(
        const Size2& resolution,
        const Vector2D& gridSpacing = Vector2D(1.0, 1.0),
        const Vector2D& origin = Vector2D(),
        const Vector2D& initialValue = Vector2D());

    //! Copy constructor.
    VertexCenteredVectorGrid2(const VertexCenteredVectorGrid2& other);

    //! Returns the actual data point size.
    Size2 dataSize() const override;

    //! Returns data position for the grid point at (0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector2D dataOrigin() const override;

    //!
    //! \brief Swaps the contents with the given \p other grid.
    //!
    //! This function swaps the contents of the grid instance with the given
    //! grid object \p other only if \p other has the same type with this grid.
    //!
    void swap(Grid2* other) override;

    //! Sets the contents with the given \p other grid.
    void set(const VertexCenteredVectorGrid2& other);

    //! Sets the contents with the given \p other grid.
    VertexCenteredVectorGrid2& operator=(
        const VertexCenteredVectorGrid2& other);

    //! Fills the grid with given value.
    void fill(const Vector2D& value) override;

    //! Fills the grid with given function.
    void fill(const std::function<Vector2D(const Vector2D&)>& func) override;

    //! Returns the copy of the grid instance.
    std::shared_ptr<VectorGrid2> clone() const override;

    //! Returns the grid builder instance.
    static VectorGridBuilder2Ptr builder();
};

//! A grid builder class that returns 2-D vertex-centered vector grid.
class VertexCenteredVectorGridBuilder2 final : public VectorGridBuilder2 {
 public:
    //! Returns a cell-centered grid for given parameters.
    VectorGrid2Ptr build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        const Vector2D& initialVal) const override {
        return std::make_shared<VertexCenteredVectorGrid2>(
            resolution,
            gridSpacing,
            gridOrigin,
            initialVal);
    }
};

}  // namespace jet

#endif  // INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID2_H_
