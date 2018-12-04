// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_H_
#define INCLUDE_JET_GRID_H_

#include <jet/bounding_box.h>
#include <jet/matrix.h>
#include <jet/serialization.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace jet {

template <size_t N>
class GridDataPositionFunc final {
 public:
    using RawFunctionType =
        std::function<Vector<double, N>(const Vector<size_t, N>&)>;

    GridDataPositionFunc(const RawFunctionType& func) : _func(func) {}

    template <typename... Indices>
    Vector<double, N> operator()(size_t i, Indices... indices) const {
        return (*this)(Vector<size_t, N>(i, indices...));
    }

    Vector<double, N> operator()(const Vector<size_t, N>& idx) const {
        return _func(idx);
    }

 private:
    RawFunctionType _func;
};

//!
//! \brief Abstract base class for N-D cartesian grid structure.
//!
//! This class represents N-D cartesian grid structure. This class is an
//! abstract base class and does not store any data. The class only stores the
//! shape of the grid. The grid structure is axis-aligned and can have different
//! grid spacing per axis.
//!
template <size_t N>
class Grid : public Serializable {
 public:
    //! Constructs an empty grid.
    Grid();

    //! Default destructor.
    virtual ~Grid();

    //! Returns the type name of derived grid.
    virtual std::string typeName() const = 0;

    //! Returns the grid resolution.
    const Vector<size_t, N>& resolution() const;

    //! Returns the grid origin.
    const Vector<double, N>& origin() const;

    //! Returns the grid spacing.
    const Vector<double, N>& gridSpacing() const;

    //! Returns the bounding box of the grid.
    const BoundingBox<double, N>& boundingBox() const;

    //! Returns the function that maps grid index to the cell-center position.
    GridDataPositionFunc<N> cellCenterPosition() const;

    //!
    //! \brief Invokes the given function \p func for each grid cell.
    //!
    //! This function invokes the given function object \p func for each grid
    //! cell in serial manner. The input parameters are i, j (and k for 3-D)
    //! indices of a grid cell. The order of execution is i-first, j-next.
    //!
    void forEachCellIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const;

    template <size_t M = N>
    std::enable_if_t<M == 2, void> forEachCellIndex(
        const std::function<void(size_t, size_t)>& func) const {
        forEachCellIndex([&func](const Vector2UZ& idx) { func(idx.x, idx.y); });
    }

    template <size_t M = N>
    std::enable_if_t<M == 3, void> forEachCellIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const {
        forEachCellIndex(
            [&func](const Vector3UZ& idx) { func(idx.x, idx.y, idx.z); });
    }

    //!
    //! \brief Invokes the given function \p func for each grid cell in
    //! parallel.
    //!
    //! This function invokes the given function object \p func for each grid
    //! cell in parallel manner. The input parameters are i, j (and k for 3-D)
    //! indices of a grid cell. The order of execution can be arbitrary since
    //! it's multi-threaded.
    //!
    void parallelForEachCellIndex(
        const std::function<void(const Vector<size_t, N>&)>& func) const;

    template <size_t M = N>
    std::enable_if_t<M == 2, void> parallelForEachCellIndex(
        const std::function<void(size_t, size_t)>& func) const {
        parallelForEachCellIndex(
            [&func](const Vector2UZ& idx) { func(idx.x, idx.y); });
    }

    template <size_t M = N>
    std::enable_if_t<M == 3, void> parallelForEachCellIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const {
        parallelForEachCellIndex(
            [&func](const Vector3UZ& idx) { func(idx.x, idx.y, idx.z); });
    }

    //! Returns true if resolution, grid-spacing and origin are same.
    bool hasSameShape(const Grid& other) const;

    //! Swaps the data with other grid.
    virtual void swap(Grid* other) = 0;

 protected:
    //! Sets the size parameters including the resolution, grid spacing, and
    //! origin.
    void setSizeParameters(const Vector<size_t, N>& resolution,
                           const Vector<double, N>& gridSpacing,
                           const Vector<double, N>& origin);

    //! Swaps the size parameters with given grid \p other.
    void swapGrid(Grid* other);

    //! Sets the size parameters with given grid \p other.
    void setGrid(const Grid& other);

    //! Fetches the data into a continuous linear array.
    virtual void getData(Array1<double>& data) const = 0;

    //! Sets the data from a continuous linear array.
    virtual void setData(const ConstArrayView1<double>& data) = 0;

 private:
    // parentheses around some of the initialization expressions due to:
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=52595
    Vector<size_t, N> _resolution;
    Vector<double, N> _gridSpacing = (Vector<double, N>::makeConstant(1));
    Vector<double, N> _origin;
    BoundingBox<double, N> _boundingBox =
        (BoundingBox<double, N>(Vector<double, N>(), Vector<double, N>()));
};

//! 2-D Grid type.
using Grid2 = Grid<2>;

//! 3-D Grid type.
using Grid3 = Grid<3>;

//! Shared pointer type for Grid.
using Grid2Ptr = std::shared_ptr<Grid2>;

//! Shared pointer type for Grid3.
using Grid3Ptr = std::shared_ptr<Grid3>;

#define JET_GRID_TYPE_NAME(DerivedClassName, N)       \
    std::string typeName() const override {           \
        return #DerivedClassName + std::to_string(N); \
    }

}  // namespace jet

#endif  // INCLUDE_JET_GRID_H_
