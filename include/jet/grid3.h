// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID3_H_
#define INCLUDE_JET_GRID3_H_

#include <jet/size3.h>
#include <jet/bounding_box3.h>
#include <functional>
#include <utility>  // just make cpplint happy..

namespace jet {

//!
//! \brief Abstract base class for 3-D cartesian grid structure.
//!
class Grid3 {
 public:
    typedef std::function<Vector3D(size_t, size_t, size_t)> DataPositionFunc;

    Grid3();

    virtual ~Grid3();

    const Size3& resolution() const;

    const Vector3D& origin() const;

    const Vector3D& gridSpacing() const;

    const BoundingBox3D& boundingBox() const;

    DataPositionFunc cellCenterPosition() const;

    void forEachCellIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    void parallelForEachCellIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //! Serializes the grid instance to the output stream \p strm.
    virtual void serialize(std::ostream* strm) const = 0;

    //! Deserializes the input stream \p strm to the grid instance.
    virtual void deserialize(std::istream* strm) = 0;

    //! Returns true if resolution, grid-spacing and origin are same.
    bool hasSameShape(const Grid3& other) const;

    //! Swaps the data with other grid.
    virtual void swap(Grid3* other) = 0;

 protected:
    void setSizeParameters(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& origin);

    void swapGrid(Grid3* other);

    void setGrid(const Grid3& other);

    void serializeGrid(std::ostream* strm) const;

    void deserializeGrid(std::istream* strm);

 private:
    Size3 _resolution;
    Vector3D _gridSpacing = Vector3D(1, 1, 1);
    Vector3D _origin;
    BoundingBox3D _boundingBox = BoundingBox3D(Vector3D(), Vector3D());
};

}  // namespace jet

#endif  // INCLUDE_JET_GRID3_H_
