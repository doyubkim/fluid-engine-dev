// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SCALAR_GRID3_H_
#define INCLUDE_JET_SCALAR_GRID3_H_

#include <jet/array3.h>
#include <jet/array_accessor3.h>
#include <jet/array_samplers3.h>
#include <jet/grid3.h>
#include <jet/scalar_field3.h>
#include <memory>
#include <vector>

namespace jet {

//! Abstract base class for 3-D scalar grid structure.
class ScalarGrid3 : public ScalarField3, public Grid3 {
 public:
    //! Read-write array accessor type.
    typedef ArrayAccessor3<double> ScalarDataAccessor;

    //! Read-only array accessor type.
    typedef ConstArrayAccessor3<double> ConstScalarDataAccessor;

    //! Constructs an empty grid.
    ScalarGrid3();

    //! Default destructor.
    virtual ~ScalarGrid3();

    //!
    //! \brief Returns the size of the grid data.
    //!
    //! This function returns the size of the grid data which is not necessarily
    //! equal to the grid resolution if the data is not stored at cell-center.
    //!
    virtual Size3 dataSize() const = 0;

    //!
    //! \brief Returns the origin of the grid data.
    //!
    //! This function returns data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    virtual Vector3D dataOrigin() const = 0;

    //! Returns the copy of the grid instance.
    virtual std::shared_ptr<ScalarGrid3> clone() const = 0;

    //! Clears the contents of the grid.
    void clear();

    //! Resizes the grid using given parameters.
    void resize(
        size_t resolutionX,
        size_t resolutionY,
        size_t resolutionZ,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double gridSpacingZ = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double originZ = 0.0,
        double initialValue = 0.0);

    //! Resizes the grid using given parameters.
    void resize(
        const Size3& resolution,
        const Vector3D& gridSpacing = Vector3D(1, 1, 1),
        const Vector3D& origin = Vector3D(),
        double initialValue = 0.0);

    //! Resizes the grid using given parameters.
    void resize(
        double gridSpacingX,
        double gridSpacingY,
        double gridSpacingZ,
        double originX,
        double originY,
        double originZ);

    //! Resizes the grid using given parameters.
    void resize(const Vector3D& gridSpacing, const Vector3D& origin);

    //! Returns the grid data at given data point.
    const double& operator()(size_t i, size_t j, size_t k) const;

    //! Returns the grid data at given data point.
    double& operator()(size_t i, size_t j, size_t k);

    //! Returns the gradient vector at given data point.
    Vector3D gradientAtDataPoint(size_t i, size_t j, size_t k) const;

    //! Returns the Laplacian at given data point.
    double laplacianAtDataPoint(size_t i, size_t j, size_t k) const;

    //! Returns the read-write data array accessor.
    ScalarDataAccessor dataAccessor();

    //! Returns the read-only data array accessor.
    ConstScalarDataAccessor constDataAccessor() const;

    //! Returns the function that maps data point to its position.
    DataPositionFunc dataPosition() const;

    //! Fills the grid with given value.
    void fill(double value,
              ExecutionPolicy policy = ExecutionPolicy::kParallel);

    //! Fills the grid with given position-to-value mapping function.
    void fill(const std::function<double(const Vector3D&)>& func,
              ExecutionPolicy policy = ExecutionPolicy::kParallel);

    //!
    //! \brief Invokes the given function \p func for each data point.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in serial manner. The input parameters are i and j indices of a
    //! data point. The order of execution is i-first, j-last.
    //!
    void forEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each data point
    //! parallelly.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in parallel manner. The input parameters are i and j indices of a
    //! data point. The order of execution can be arbitrary since it's
    //! multi-threaded.
    //!
    void parallelForEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    // ScalarField3 implementations

    //!
    //! \brief Returns the sampled value at given position \p x.
    //!
    //! This function returns the data sampled at arbitrary position \p x.
    //! The sampling function is linear.
    //!
    double sample(const Vector3D& x) const override;

    //!
    //! \brief Returns the sampler function.
    //!
    //! This function returns the data sampler function object. The sampling
    //! function is linear.
    //!
    std::function<double(const Vector3D&)> sampler() const override;

    //! Returns the gradient vector at given position \p x.
    Vector3D gradient(const Vector3D& x) const override;

    //! Returns the Laplacian at given position \p x.
    double laplacian(const Vector3D& x) const override;

    //! Serializes the grid instance to the output buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the input buffer to the grid instance.
    void deserialize(const std::vector<uint8_t>& buffer) override;

 protected:
    //! Swaps the data storage and predefined samplers with given grid.
    void swapScalarGrid(ScalarGrid3* other);

    //! Sets the data storage and predefined samplers with given grid.
    void setScalarGrid(const ScalarGrid3& other);

    //! Fetches the data into a continuous linear array.
    void getData(std::vector<double>* data) const override;

    //! Sets the data from a continuous linear array.
    void setData(const std::vector<double>& data) override;

 private:
    Array3<double> _data;
    LinearArraySampler3<double, double> _linearSampler;
    std::function<double(const Vector3D&)> _sampler;

    void resetSampler();
};

//! Shared pointer for the ScalarGrid3 type.
typedef std::shared_ptr<ScalarGrid3> ScalarGrid3Ptr;

//! Abstract base class for 3-D scalar grid builder.
class ScalarGridBuilder3 {
 public:
    //! Creates a builder.
    ScalarGridBuilder3();

    //! Default destructor.
    virtual ~ScalarGridBuilder3();

    //! Returns 3-D scalar grid with given parameters.
    virtual ScalarGrid3Ptr build(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin,
        double initialVal) const = 0;
};

//! Shared pointer for the ScalarGridBuilder3 type.
typedef std::shared_ptr<ScalarGridBuilder3> ScalarGridBuilder3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_GRID3_H_
