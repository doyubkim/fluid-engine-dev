// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SCALAR_GRID2_H_
#define INCLUDE_JET_SCALAR_GRID2_H_

#include <jet/array2.h>
#include <jet/array_accessor2.h>
#include <jet/array_samplers2.h>
#include <jet/grid2.h>
#include <jet/scalar_field2.h>
#include <memory>
#include <vector>

namespace jet {

//! Abstract base class for 2-D scalar grid structure.
class ScalarGrid2 : public ScalarField2, public Grid2 {
 public:
    //! Read-write array accessor type.
    typedef ArrayAccessor2<double> ScalarDataAccessor;

    //! Read-only array accessor type.
    typedef ConstArrayAccessor2<double> ConstScalarDataAccessor;

    //! Constructs an empty grid.
    ScalarGrid2();

    //! Default destructor.
    virtual ~ScalarGrid2();

    //!
    //! \brief Returns the size of the grid data.
    //!
    //! This function returns the size of the grid data which is not necessarily
    //! equal to the grid resolution if the data is not stored at cell-center.
    //!
    virtual Size2 dataSize() const = 0;

    //!
    //! \brief Returns the origin of the grid data.
    //!
    //! This function returns data position for the grid point at (0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    virtual Vector2D dataOrigin() const = 0;

    //! Returns the copy of the grid instance.
    virtual std::shared_ptr<ScalarGrid2> clone() const = 0;

    //! Clears the contents of the grid.
    void clear();

    //! Resizes the grid using given parameters.
    void resize(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double initialValue = 0.0);

    //! Resizes the grid using given parameters.
    void resize(
        const Size2& resolution,
        const Vector2D& gridSpacing = Vector2D(1, 1),
        const Vector2D& origin = Vector2D(),
        double initialValue = 0.0);

    //! Resizes the grid using given parameters.
    void resize(
        double gridSpacingX,
        double gridSpacingY,
        double originX,
        double originY);

    //! Resizes the grid using given parameters.
    void resize(const Vector2D& gridSpacing, const Vector2D& origin);

    //! Returns the grid data at given data point.
    const double& operator()(size_t i, size_t j) const;

    //! Returns the grid data at given data point.
    double& operator()(size_t i, size_t j);

    //! Returns the gradient vector at given data point.
    Vector2D gradientAtDataPoint(size_t i, size_t j) const;

    //! Returns the Laplacian at given data point.
    double laplacianAtDataPoint(size_t i, size_t j) const;

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
    void fill(const std::function<double(const Vector2D&)>& func,
              ExecutionPolicy policy = ExecutionPolicy::kParallel);

    //!
    //! \brief Invokes the given function \p func for each data point.
    //!
    //! This function invokes the given function object \p func for each data
    //! point in serial manner. The input parameters are i and j indices of a
    //! data point. The order of execution is i-first, j-last.
    //!
    void forEachDataPointIndex(
        const std::function<void(size_t, size_t)>& func) const;

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
        const std::function<void(size_t, size_t)>& func) const;

    // ScalarField2 implementations

    //!
    //! \brief Returns the sampled value at given position \p x.
    //!
    //! This function returns the data sampled at arbitrary position \p x.
    //! The sampling function is linear.
    //!
    double sample(const Vector2D& x) const override;

    //!
    //! \brief Returns the sampler function.
    //!
    //! This function returns the data sampler function object. The sampling
    //! function is linear.
    //!
    std::function<double(const Vector2D&)> sampler() const override;

    //! Returns the gradient vector at given position \p x.
    Vector2D gradient(const Vector2D& x) const override;

    //! Returns the Laplacian at given position \p x.
    double laplacian(const Vector2D& x) const override;

    //! Serializes the grid instance to the output buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the input buffer to the grid instance.
    void deserialize(const std::vector<uint8_t>& buffer) override;

 protected:
    //! Swaps the data storage and predefined samplers with given grid.
    void swapScalarGrid(ScalarGrid2* other);

    //! Sets the data storage and predefined samplers with given grid.
    void setScalarGrid(const ScalarGrid2& other);

    //! Fetches the data into a continuous linear array.
    void getData(std::vector<double>* data) const override;

    //! Sets the data from a continuous linear array.
    void setData(const std::vector<double>& data) override;

 private:
    Array2<double> _data;
    LinearArraySampler2<double, double> _linearSampler;
    std::function<double(const Vector2D&)> _sampler;

    void resetSampler();
};

//! Shared pointer for the ScalarGrid2 type.
typedef std::shared_ptr<ScalarGrid2> ScalarGrid2Ptr;

//! Abstract base class for 2-D scalar grid builder.
class ScalarGridBuilder2 {
 public:
    //! Creates a builder.
    ScalarGridBuilder2();

    //! Default destructor.
    virtual ~ScalarGridBuilder2();

    //! Returns 2-D scalar grid with given parameters.
    virtual ScalarGrid2Ptr build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        double initialVal) const = 0;
};

//! Shared pointer for the ScalarGridBuilder2 type.
typedef std::shared_ptr<ScalarGridBuilder2> ScalarGridBuilder2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_GRID2_H_
