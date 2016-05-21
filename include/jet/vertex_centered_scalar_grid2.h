// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID2_H_
#define INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID2_H_

#include <jet/array2.h>
#include <jet/scalar_grid2.h>
#include <algorithm>  // just make cpplint happy..

namespace jet {

class VertexCenteredScalarGrid2 final : public ScalarGrid2 {
 public:
    VertexCenteredScalarGrid2();

    explicit VertexCenteredScalarGrid2(
         size_t resolutionX,
         size_t resolutionY,
         double gridSpacingX = 1.0,
         double gridSpacingY = 1.0,
         double originX = 0.0,
         double originY = 0.0,
         double initialValue = 0.0);

    explicit VertexCenteredScalarGrid2(
        const Size2& resolution,
        const Vector2D& gridSpacing = Vector2D(1.0, 1.0),
        const Vector2D& origin = Vector2D(),
        double initialValue = 0.0);

    VertexCenteredScalarGrid2(const VertexCenteredScalarGrid2& other);

    virtual ~VertexCenteredScalarGrid2();

    Size2 dataSize() const override;

    //! Returns data position for the grid point at (0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector2D dataOrigin() const override;

    std::shared_ptr<ScalarGrid2> clone() const override;

    void swap(Grid2* other) override;

    void set(const VertexCenteredScalarGrid2& other);

    VertexCenteredScalarGrid2& operator=(
        const VertexCenteredScalarGrid2& other);

    static ScalarGridBuilder2Ptr builder();
};


class VertexCenteredScalarGridBuilder2 final : public ScalarGridBuilder2 {
 public:
    VertexCenteredScalarGridBuilder2();

    ScalarGrid2Ptr build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        double initialVal) const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID2_H_
