// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CELL_CENTERED_VECTOR_GRID2_H_
#define INCLUDE_JET_CELL_CENTERED_VECTOR_GRID2_H_

#include <jet/array2.h>
#include <jet/collocated_vector_grid2.h>

namespace jet {

class CellCenteredVectorGrid2 final : public CollocatedVectorGrid2 {
 public:
    CellCenteredVectorGrid2();

    explicit CellCenteredVectorGrid2(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double initialValueU = 0.0,
        double initialValueV = 0.0);

    explicit CellCenteredVectorGrid2(
        const Size2& resolution,
        const Vector2D& gridSpacing = Vector2D(1.0, 1.0),
        const Vector2D& origin = Vector2D(),
        const Vector2D& initialValue = Vector2D());

    CellCenteredVectorGrid2(const CellCenteredVectorGrid2& other);

    Size2 dataSize() const override;

    //! Returns data position for the grid point at (0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector2D dataOrigin() const override;

    void swap(Grid2* other) override;

    void set(const CellCenteredVectorGrid2& other);

    CellCenteredVectorGrid2& operator=(const CellCenteredVectorGrid2& other);

    void fill(const Vector2D& value) override;

    void fill(const std::function<Vector2D(const Vector2D&)>& func) override;

    std::shared_ptr<VectorGrid2> clone() const override;

    static VectorGridBuilder2Ptr builder();
};


class CellCenteredVectorGridBuilder2 final : public VectorGridBuilder2 {
 public:
    CellCenteredVectorGridBuilder2();

    VectorGrid2Ptr build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        const Vector2D& initialVal) const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_CELL_CENTERED_VECTOR_GRID2_H_
