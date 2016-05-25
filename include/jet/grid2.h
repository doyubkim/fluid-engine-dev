// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID2_H_
#define INCLUDE_JET_GRID2_H_

#include <jet/size2.h>
#include <jet/bounding_box2.h>
#include <functional>
#include <utility>  // just make cpplint happy..

namespace jet {

//!
//! \brief Abstract base class for 2-D cartesian grid structure.
//!
class Grid2 {
 public:
    typedef std::function<Vector2D(size_t, size_t)> DataPositionFunc;

    Grid2();

    virtual ~Grid2();

    const Size2& resolution() const;

    const Vector2D& origin() const;

    const Vector2D& gridSpacing() const;

    const BoundingBox2D& boundingBox() const;

    DataPositionFunc cellCenterPosition() const;

    void forEachCellIndex(
        const std::function<void(size_t, size_t)>& func) const;

    void parallelForEachCellIndex(
        const std::function<void(size_t, size_t)>& func) const;

    //! Serializes the grid instance to the output stream \p strm.
    virtual void serialize(std::ostream* strm) const = 0;

    //! Deserializes the input stream \p strm to the grid instance.
    virtual void deserialize(std::istream* strm) = 0;

    //! Returns true if resolution, grid-spacing and origin are same.
    bool hasSameShape(const Grid2& other) const;

    //! Swaps the data with other grid.
    virtual void swap(Grid2* other) = 0;

 protected:
    void setSizeParameters(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& origin);

    void swapGrid(Grid2* other);

    void setGrid(const Grid2& other);

    void serializeGrid(std::ostream* strm) const;

    void deserializeGrid(std::istream* strm);

 private:
    Size2 _resolution;
    Vector2D _gridSpacing = Vector2D(1, 1);
    Vector2D _origin;
    BoundingBox2D _boundingBox = BoundingBox2D(Vector2D(), Vector2D());
};

}  // namespace jet

#endif  // INCLUDE_JET_GRID2_H_
