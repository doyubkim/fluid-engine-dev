// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FACE_CENTERED_GRID2_H_
#define INCLUDE_JET_FACE_CENTERED_GRID2_H_

#include <jet/array2.h>
#include <jet/array_samplers2.h>
#include <jet/vector_grid2.h>
#include <memory>
#include <utility>  // just make cpplint happy..
#include <vector>

namespace jet {

//!
//! \brief 2-D face-centered (a.k.a MAC or staggered) grid.
//!
//! This class implements face-centered grid which is also known as
//! marker-and-cell (MAC) or staggered grid. This vector grid stores each vector
//! component at face center. Thus, u and v components are not collocated.
//!
class FaceCenteredGrid2 final : public VectorGrid2 {
 public:
    JET_GRID2_TYPE_NAME(FaceCenteredGrid2)

    class Builder;

    //! Read-write scalar data accessor type.
    typedef ArrayAccessor2<double> ScalarDataAccessor;

    //! Read-only scalar data accessor type.
    typedef ConstArrayAccessor2<double> ConstScalarDataAccessor;

    //! Constructs empty grid.
    FaceCenteredGrid2();

    //! Resizes the grid using given parameters.
    FaceCenteredGrid2(size_t resolutionX, size_t resolutionY,
                      double gridSpacingX = 1.0, double gridSpacingY = 1.0,
                      double originX = 0.0, double originY = 0.0,
                      double initialValueU = 0.0, double initialValueV = 0.0);

    //! Resizes the grid using given parameters.
    FaceCenteredGrid2(const Size2& resolution,
                      const Vector2D& gridSpacing = Vector2D(1.0, 1.0),
                      const Vector2D& origin = Vector2D(),
                      const Vector2D& initialValue = Vector2D());

    //! Copy constructor.
    FaceCenteredGrid2(const FaceCenteredGrid2& other);

    //!
    //! \brief Swaps the contents with the given \p other grid.
    //!
    //! This function swaps the contents of the grid instance with the given
    //! grid object \p other only if \p other has the same type with this grid.
    //!
    void swap(Grid2* other) override;

    //! Sets the contents with the given \p other grid.
    void set(const FaceCenteredGrid2& other);

    //! Sets the contents with the given \p other grid.
    FaceCenteredGrid2& operator=(const FaceCenteredGrid2& other);

    //! Returns u-value at given data point.
    double& u(size_t i, size_t j);

    //! Returns u-value at given data point.
    const double& u(size_t i, size_t j) const;

    //! Returns v-value at given data point.
    double& v(size_t i, size_t j);

    //! Returns v-value at given data point.
    const double& v(size_t i, size_t j) const;

    //! Returns interpolated value at cell center.
    Vector2D valueAtCellCenter(size_t i, size_t j) const;

    //! Returns divergence at cell-center location.
    double divergenceAtCellCenter(size_t i, size_t j) const;

    //! Returns curl at cell-center location.
    double curlAtCellCenter(size_t i, size_t j) const;

    //! Returns u data accessor.
    ScalarDataAccessor uAccessor();

    //! Returns read-only u data accessor.
    ConstScalarDataAccessor uConstAccessor() const;

    //! Returns v data accessor.
    ScalarDataAccessor vAccessor();

    //! Returns read-only v data accessor.
    ConstScalarDataAccessor vConstAccessor() const;

    //! Returns function object that maps u data point to its actual position.
    DataPositionFunc uPosition() const;

    //! Returns function object that maps v data point to its actual position.
    DataPositionFunc vPosition() const;

    //! Returns data size of the u component.
    Size2 uSize() const;

    //! Returns data size of the v component.
    Size2 vSize() const;

    //!
    //! \brief Returns u-data position for the grid point at (0, 0).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    Vector2D uOrigin() const;

    //!
    //! \brief Returns v-data position for the grid point at (0, 0).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    Vector2D vOrigin() const;

    //! Fills the grid with given value.
    void fill(const Vector2D& value,
              ExecutionPolicy policy = ExecutionPolicy::kParallel) override;

    //! Fills the grid with given function.
    void fill(const std::function<Vector2D(const Vector2D&)>& func,
              ExecutionPolicy policy = ExecutionPolicy::kParallel) override;

    //! Returns the copy of the grid instance.
    std::shared_ptr<VectorGrid2> clone() const override;

    //!
    //! \brief Invokes the given function \p func for each u-data point.
    //!
    //! This function invokes the given function object \p func for each u-data
    //! point in serial manner. The input parameters are i and j indices of a
    //! u-data point. The order of execution is i-first, j-last.
    //!
    void forEachUIndex(const std::function<void(size_t, size_t)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each u-data point
    //! parallelly.
    //!
    //! This function invokes the given function object \p func for each u-data
    //! point in parallel manner. The input parameters are i and j indices of a
    //! u-data point. The order of execution can be arbitrary since it's
    //! multi-threaded.
    //!
    void parallelForEachUIndex(
        const std::function<void(size_t, size_t)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each v-data point.
    //!
    //! This function invokes the given function object \p func for each v-data
    //! point in serial manner. The input parameters are i and j indices of a
    //! v-data point. The order of execution is i-first, j-last.
    //!
    void forEachVIndex(const std::function<void(size_t, size_t)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each v-data point
    //! parallelly.
    //!
    //! This function invokes the given function object \p func for each v-data
    //! point in parallel manner. The input parameters are i and j indices of a
    //! v-data point. The order of execution can be arbitrary since it's
    //! multi-threaded.
    //!
    void parallelForEachVIndex(
        const std::function<void(size_t, size_t)>& func) const;

    // VectorField2 implementations

    //! Returns sampled value at given position \p x.
    Vector2D sample(const Vector2D& x) const override;

    //! Returns divergence at given position \p x.
    double divergence(const Vector2D& x) const override;

    //! Returns curl at given position \p x.
    double curl(const Vector2D& x) const override;

    //!
    //! \brief Returns the sampler function.
    //!
    //! This function returns the data sampler function object. The sampling
    //! function is linear.
    //!
    std::function<Vector2D(const Vector2D&)> sampler() const override;

    //! Returns builder fox FaceCenteredGrid2.
    static Builder builder();

 protected:
    // VectorGrid2 implementations
    void onResize(const Size2& resolution, const Vector2D& gridSpacing,
                  const Vector2D& origin, const Vector2D& initialValue) final;

    //! Fetches the data into a continuous linear array.
    void getData(std::vector<double>* data) const override;

    //! Sets the data from a continuous linear array.
    void setData(const std::vector<double>& data) override;

 private:
    Array2<double> _dataU;
    Array2<double> _dataV;
    Vector2D _dataOriginU;
    Vector2D _dataOriginV;
    LinearArraySampler2<double, double> _uLinearSampler;
    LinearArraySampler2<double, double> _vLinearSampler;
    std::function<Vector2D(const Vector2D&)> _sampler;

    void resetSampler();
};

//! Shared pointer type for the FaceCenteredGrid2.
typedef std::shared_ptr<FaceCenteredGrid2> FaceCenteredGrid2Ptr;

//!
//! \brief Front-end to create FaceCenteredGrid2 objects step by step.
//!
class FaceCenteredGrid2::Builder final : public VectorGridBuilder2 {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Size2& resolution);

    //! Returns builder with resolution.
    Builder& withResolution(size_t resolutionX, size_t resolutionY);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(const Vector2D& gridSpacing);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(double gridSpacingX, double gridSpacingY);

    //! Returns builder with grid origin.
    Builder& withOrigin(const Vector2D& gridOrigin);

    //! Returns builder with grid origin.
    Builder& withOrigin(double gridOriginX, double gridOriginY);

    //! Returns builder with initial value.
    Builder& withInitialValue(const Vector2D& initialVal);

    //! Returns builder with initial value.
    Builder& withInitialValue(double initialValX, double initialValY);

    //! Builds FaceCenteredGrid2 instance.
    FaceCenteredGrid2 build() const;

    //! Builds shared pointer of FaceCenteredGrid2 instance.
    FaceCenteredGrid2Ptr makeShared() const;

    //!
    //! \brief Builds shared pointer of FaceCenteredGrid2 instance.
    //!
    //! This is an overriding function that implements VectorGridBuilder2.
    //!
    VectorGrid2Ptr build(const Size2& resolution, const Vector2D& gridSpacing,
                         const Vector2D& gridOrigin,
                         const Vector2D& initialVal) const override;

 private:
    Size2 _resolution{1, 1};
    Vector2D _gridSpacing{1, 1};
    Vector2D _gridOrigin{0, 0};
    Vector2D _initialVal{0, 0};
};

}  // namespace jet

#endif  // INCLUDE_JET_FACE_CENTERED_GRID2_H_
