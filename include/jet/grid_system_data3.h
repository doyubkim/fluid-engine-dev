// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_SYSTEM_DATA3_H_
#define INCLUDE_JET_GRID_SYSTEM_DATA3_H_

#include <jet/face_centered_grid3.h>
#include <jet/scalar_grid3.h>
#include <jet/serialization.h>
#include <memory>
#include <vector>

namespace jet {

//!
//! \brief      3-D grid system data.
//!
//! This class is the key data structure for storing grid system data. To
//! represent a grid system for fluid simulation, velocity field is defined as a
//! face-centered (MAC) grid by default. It can also have additional scalar or
//! vector attributes by adding extra data layer.
//!
class GridSystemData3 : public Serializable {
 public:
    //! Constructs empty grid system.
    GridSystemData3();

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
    GridSystemData3(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& origin);

    //! Copy constructor.
    GridSystemData3(const GridSystemData3& other);

    //! Destructor.
    virtual ~GridSystemData3();

    //!
    //! \brief      Resizes the whole system with given resolution, grid
    //!             spacing, and origin.
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
    void resize(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& origin);

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
    Size3 resolution() const;

    //! Return the grid spacing.
    Vector3D gridSpacing() const;

    //! Returns the origin of the grid.
    Vector3D origin() const;

    //! Returns the bounding box of the grid.
    BoundingBox3D boundingBox() const;

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
    size_t addScalarData(
        const ScalarGridBuilder3Ptr& builder,
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
        const VectorGridBuilder3Ptr& builder,
        const Vector3D& initialVal = Vector3D());

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
        const ScalarGridBuilder3Ptr& builder,
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
        const VectorGridBuilder3Ptr& builder,
        const Vector3D& initialVal = Vector3D());

    //!
    //! \brief      Returns the velocity field.
    //!
    //! This class has velocify field by default, and it is part of the
    //! advectable vector data list.
    //!
    //! \return     Pointer to the velocity field.
    //!
    const FaceCenteredGrid3Ptr& velocity() const;

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
    const ScalarGrid3Ptr& scalarDataAt(size_t idx) const;

    //! Returns the non-advectable vector data at given index.
    const VectorGrid3Ptr& vectorDataAt(size_t idx) const;

    //! Returns the advectable scalar data at given index.
    const ScalarGrid3Ptr& advectableScalarDataAt(size_t idx) const;

    //! Returns the advectable vector data at given index.
    const VectorGrid3Ptr& advectableVectorDataAt(size_t idx) const;

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
    Size3 _resolution;
    Vector3D _gridSpacing;
    Vector3D _origin;

    FaceCenteredGrid3Ptr _velocity;
    size_t _velocityIdx;
    std::vector<ScalarGrid3Ptr> _scalarDataList;
    std::vector<VectorGrid3Ptr> _vectorDataList;
    std::vector<ScalarGrid3Ptr> _advectableScalarDataList;
    std::vector<VectorGrid3Ptr> _advectableVectorDataList;
};

//! Shared pointer type of GridSystemData3.
typedef std::shared_ptr<GridSystemData3> GridSystemData3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SYSTEM_DATA3_H_
