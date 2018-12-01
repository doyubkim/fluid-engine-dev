// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_SYSTEM_DATA_H_
#define INCLUDE_JET_GRID_SYSTEM_DATA_H_

#include <jet/face_centered_grid.h>
#include <jet/scalar_grid.h>
#include <jet/serialization.h>

namespace jet {

//!
//! \brief      N-D grid system data.
//!
//! This class is the key data structure for storing grid system data. To
//! represent a grid system for fluid simulation, velocity field is defined as a
//! face-centered (MAC) grid by default. It can also have additional scalar or
//! vector attributes by adding extra data layer.
//!
template <size_t N>
class GridSystemData : public Serializable {
 public:
    //! Constructs empty grid system.
    GridSystemData();

    //!
    //! \brief      Constructs a grid system with given resolution, grid spacing
    //!             and origin.
    //!
    //! This constructor builds the entire grid layers within the system. Note,
    //! the resolution is the grid resolution, not the data size of each grid.
    //! Depending on the layout of the grid, the data point may lie on different
    //! part of the grid (vertex, cell-center, or face-center), thus can have
    //! different array size internally. The resolution of the grid means the
    //! grid cell resolution.
    //!
    //! \param[in]  resolution  The resolution.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  origin      The origin.
    //!
    GridSystemData(const Vector<size_t, N>& resolution,
                   const Vector<double, N>& gridSpacing,
                   const Vector<double, N>& origin);

    //! Copy constructor.
    GridSystemData(const GridSystemData& other);

    //! Destructor.
    virtual ~GridSystemData();

    //!
    //! \brief      Resizes the whole system with given resolution, grid
    //!     spacing, and origin.
    //!
    //! This function resizes the entire grid layers within the system. Note,
    //! the resolution is the grid resolution, not the data size of each grid.
    //! Depending on the layout of the grid, the data point may lie on different
    //! part of the grid (vertex, cell-center, or face-center), thus can have
    //! different array size internally. The resolution of the grid means the
    //! grid cell resolution.
    //!
    //! \param[in]  resolution  The resolution.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  origin      The origin.
    //!
    void resize(const Vector<size_t, N>& resolution,
                const Vector<double, N>& gridSpacing,
                const Vector<double, N>& origin);

    //!
    //! \brief      Returns the resolution of the grid.
    //!
    //! This function resizes the entire grid layers within the system. Note,
    //! the resolution is the grid resolution, not the data size of each grid.
    //! Depending on the layout of the grid, the data point may lie on different
    //! part of the grid (vertex, cell-center, or face-center), thus can have
    //! different array size internally. The resolution of the grid means the
    //! grid cell resolution.
    //!
    //! \return     Grid cell resolution.
    //!
    Vector<size_t, N> resolution() const;

    //! Return the grid spacing.
    Vector<double, N> gridSpacing() const;

    //! Returns the origin of the grid.
    Vector<double, N> origin() const;

    //! Returns the bounding box of the grid.
    BoundingBox<double, N> boundingBox() const;

    //!
    //! \brief      Adds a non-advectable scalar data grid by passing its
    //!     builder and initial value.
    //!
    //! This function adds a new scalar data grid. This layer is not advectable,
    //! meaning that during the computation of fluid flow, this layer won't
    //! follow the flow. For the future access of this layer, its index is
    //! returned.
    //!
    //! \param[in]  builder    The grid builder.
    //! \param[in]  initialVal The initial value.
    //!
    //! \return     Index of the data.
    //!
    size_t addScalarData(const std::shared_ptr<ScalarGridBuilder<N>>& builder,
                         double initialVal = 0.0);

    //!
    //! \brief      Adds a non-advectable vector data grid by passing its
    //!     builder and initial value.
    //!
    //! This function adds a new vector data grid. This layer is not advectable,
    //! meaning that during the computation of fluid flow, this layer won't
    //! follow the flow. For the future access of this layer, its index is
    //! returned.
    //!
    //! \param[in]  builder    The grid builder.
    //! \param[in]  initialVal The initial value.
    //!
    //! \return     Index of the data.
    //!
    size_t addVectorData(
        const std::shared_ptr<VectorGridBuilder<N>>& builder,
        const Vector<double, N>& initialVal = Vector<double, N>());

    //!
    //! \brief      Adds an advectable scalar data grid by passing its builder
    //!     and initial value.
    //!
    //! This function adds a new scalar data grid. This layer is advectable,
    //! meaning that during the computation of fluid flow, this layer will
    //! follow the flow. For the future access of this layer, its index is
    //! returned.
    //!
    //! \param[in]  builder    The grid builder.
    //! \param[in]  initialVal The initial value.
    //!
    //! \return     Index of the data.
    //!
    size_t addAdvectableScalarData(
        const std::shared_ptr<ScalarGridBuilder<N>>& builder,
        double initialVal = 0.0);

    //!
    //! \brief      Adds an advectable vector data grid by passing its builder
    //!     and initial value.
    //!
    //! This function adds a new vector data grid. This layer is advectable,
    //! meaning that during the computation of fluid flow, this layer will
    //! follow the flow. For the future access of this layer, its index is
    //! returned.
    //!
    //! \param[in]  builder    The grid builder.
    //! \param[in]  initialVal The initial value.
    //!
    //! \return     Index of the data.
    //!
    size_t addAdvectableVectorData(
        const std::shared_ptr<VectorGridBuilder<N>>& builder,
        const Vector<double, N>& initialVal = Vector<double, N>());

    //!
    //! \brief      Returns the velocity field.
    //!
    //! This class has velocify field by default, and it is part of the
    //! advectable vector data list.
    //!
    //! \return     Pointer to the velocity field.
    //!
    const std::shared_ptr<FaceCenteredGrid<N>>& velocity() const;

    //!
    //! \brief      Returns the index of the velocity field.
    //!
    //! This class has velocify field by default, and it is part of the
    //! advectable vector data list. This function returns the index of the
    //! velocity field from the list.
    //!
    //! \return     Index of the velocity field.
    //!
    size_t velocityIndex() const;

    //! Returns the non-advectable scalar data at given index.
    const std::shared_ptr<ScalarGrid<N>>& scalarDataAt(size_t idx) const;

    //! Returns the non-advectable vector data at given index.
    const std::shared_ptr<VectorGrid<N>>& vectorDataAt(size_t idx) const;

    //! Returns the advectable scalar data at given index.
    const std::shared_ptr<ScalarGrid<N>>& advectableScalarDataAt(
        size_t idx) const;

    //! Returns the advectable vector data at given index.
    const std::shared_ptr<VectorGrid<N>>& advectableVectorDataAt(
        size_t idx) const;

    //! Returns the number of non-advectable scalar data.
    size_t numberOfScalarData() const;

    //! Returns the number of non-advectable vector data.
    size_t numberOfVectorData() const;

    //! Returns the number of advectable scalar data.
    size_t numberOfAdvectableScalarData() const;

    //! Returns the number of advectable vector data.
    size_t numberOfAdvectableVectorData() const;

    //! Serialize the data to the given buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Serialize the data from the given buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

 private:
    Vector<size_t, N> _resolution;
    Vector<double, N> _gridSpacing;
    Vector<double, N> _origin;

    std::shared_ptr<FaceCenteredGrid<N>> _velocity;
    size_t _velocityIdx;
    std::vector<std::shared_ptr<ScalarGrid<N>>> _scalarDataList;
    std::vector<std::shared_ptr<VectorGrid<N>>> _vectorDataList;
    std::vector<std::shared_ptr<ScalarGrid<N>>> _advectableScalarDataList;
    std::vector<std::shared_ptr<VectorGrid<N>>> _advectableVectorDataList;

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> serialize(
        const GridSystemData<2>& grid, std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> serialize(
        const GridSystemData<3>& grid, std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> deserialize(
        const std::vector<uint8_t>& buffer, GridSystemData<2>& grid);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> deserialize(
        const std::vector<uint8_t>& buffer, GridSystemData<3>& grid);
};

//! 2-D GridSystemData type.
using GridSystemData2 = GridSystemData<2>;

//! 3-D GridSystemData type.
using GridSystemData3 = GridSystemData<3>;

//! Shared pointer type of GridSystemData2.
using GridSystemData2Ptr = std::shared_ptr<GridSystemData2>;

//! Shared pointer type of GridSystemData3.
using GridSystemData3Ptr = std::shared_ptr<GridSystemData3>;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SYSTEM_DATA_H_
