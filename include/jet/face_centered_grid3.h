// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FACE_CENTERED_GRID3_H_
#define INCLUDE_JET_FACE_CENTERED_GRID3_H_

#include <jet/array3.h>
#include <jet/array_samplers3.h>
#include <jet/vector_grid3.h>
#include <memory>
#include <utility>  // just make cpplint happy..
#include <vector>

namespace jet {

//!
//! \brief 3-D face-centered (a.k.a MAC or staggered) grid.
//!
//! This class implements face-centered grid which is also known as
//! marker-and-cell (MAC) or staggered grid. This vector grid stores each vector
//! component at face center. Thus, u, v, and w components are not collocated.
//!
class FaceCenteredGrid3 final : public VectorGrid3 {
 public:
    JET_GRID3_TYPE_NAME(FaceCenteredGrid3)

    class Builder;

    //! Read-write scalar data accessor type.
    typedef ArrayAccessor3<double> ScalarDataAccessor;

    //! Read-only scalar data accessor type.
    typedef ConstArrayAccessor3<double> ConstScalarDataAccessor;

    //! Constructs empty grid.
    FaceCenteredGrid3();

    //! Resizes the grid using given parameters.
    FaceCenteredGrid3(size_t resolutionX, size_t resolutionY,
                      size_t resolutionZ, double gridSpacingX = 1.0,
                      double gridSpacingY = 1.0, double gridSpacingZ = 1.0,
                      double originX = 0.0, double originY = 0.0,
                      double originZ = 0.0, double initialValueU = 0.0,
                      double initialValueV = 0.0, double initialValueW = 0.0);

    //! Resizes the grid using given parameters.
    FaceCenteredGrid3(const Size3& resolution,
                      const Vector3D& gridSpacing = Vector3D(1.0, 1.0, 1.0),
                      const Vector3D& origin = Vector3D(),
                      const Vector3D& initialValue = Vector3D());

    //! Copy constructor.
    FaceCenteredGrid3(const FaceCenteredGrid3& other);

    //!
    //! \brief Swaps the contents with the given \p other grid.
    //!
    //! This function swaps the contents of the grid instance with the given
    //! grid object \p other only if \p other has the same type with this grid.
    //!
    void swap(Grid3* other) override;

    //! Sets the contents with the given \p other grid.
    void set(const FaceCenteredGrid3& other);

    //! Sets the contents with the given \p other grid.
    FaceCenteredGrid3& operator=(const FaceCenteredGrid3& other);

    //! Returns u-value at given data point.
    double& u(size_t i, size_t j, size_t k);

    //! Returns u-value at given data point.
    const double& u(size_t i, size_t j, size_t k) const;

    //! Returns v-value at given data point.
    double& v(size_t i, size_t j, size_t k);

    //! Returns v-value at given data point.
    const double& v(size_t i, size_t j, size_t k) const;

    //! Returns w-value at given data point.
    double& w(size_t i, size_t j, size_t k);

    //! Returns w-value at given data point.
    const double& w(size_t i, size_t j, size_t k) const;

    //! Returns interpolated value at cell center.
    Vector3D valueAtCellCenter(size_t i, size_t j, size_t k) const;

    //! Returns divergence at cell-center location.
    double divergenceAtCellCenter(size_t i, size_t j, size_t k) const;

    //! Returns curl at cell-center location.
    Vector3D curlAtCellCenter(size_t i, size_t j, size_t k) const;

    //! Returns u data accessor.
    ScalarDataAccessor uAccessor();

    //! Returns read-only u data accessor.
    ConstScalarDataAccessor uConstAccessor() const;

    //! Returns v data accessor.
    ScalarDataAccessor vAccessor();

    //! Returns read-only v data accessor.
    ConstScalarDataAccessor vConstAccessor() const;

    //! Returns w data accessor.
    ScalarDataAccessor wAccessor();

    //! Returns read-only w data accessor.
    ConstScalarDataAccessor wConstAccessor() const;

    //! Returns function object that maps u data point to its actual position.
    DataPositionFunc uPosition() const;

    //! Returns function object that maps v data point to its actual position.
    DataPositionFunc vPosition() const;

    //! Returns function object that maps w data point to its actual position.
    DataPositionFunc wPosition() const;

    //! Returns data size of the u component.
    Size3 uSize() const;

    //! Returns data size of the v component.
    Size3 vSize() const;

    //! Returns data size of the w component.
    Size3 wSize() const;

    //!
    //! \brief Returns u-data position for the grid point at (0, 0, 0).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    Vector3D uOrigin() const;

    //!
    //! \brief Returns v-data position for the grid point at (0, 0, 0).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    Vector3D vOrigin() const;

    //!
    //! \brief Returns w-data position for the grid point at (0, 0, 0).
    //!
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    //!
    Vector3D wOrigin() const;

    //! Fills the grid with given value.
    void fill(const Vector3D& value,
              ExecutionPolicy policy = ExecutionPolicy::kParallel) override;

    //! Fills the grid with given function.
    void fill(const std::function<Vector3D(const Vector3D&)>& func,
              ExecutionPolicy policy = ExecutionPolicy::kParallel) override;

    //! Returns the copy of the grid instance.
    std::shared_ptr<VectorGrid3> clone() const override;

    //!
    //! \brief Invokes the given function \p func for each u-data point.
    //!
    //! This function invokes the given function object \p func for each u-data
    //! point in serial manner. The input parameters are i and j indices of a
    //! u-data point. The order of execution is i-first, j-last.
    //!
    void forEachUIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

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
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each v-data point.
    //!
    //! This function invokes the given function object \p func for each v-data
    //! point in serial manner. The input parameters are i and j indices of a
    //! v-data point. The order of execution is i-first, j-last.
    //!
    void forEachVIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

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
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each w-data point.
    //!
    //! This function invokes the given function object \p func for each w-data
    //! point in serial manner. The input parameters are i and j indices of a
    //! w-data point. The order of execution is i-first, j-last.
    //!
    void forEachWIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //!
    //! \brief Invokes the given function \p func for each w-data point
    //! parallelly.
    //!
    //! This function invokes the given function object \p func for each w-data
    //! point in parallel manner. The input parameters are i and j indices of a
    //! w-data point. The order of execution can be arbitrary since it's
    //! multi-threaded.
    //!
    void parallelForEachWIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    // VectorField3 implementations

    //! Returns sampled value at given position \p x.
    Vector3D sample(const Vector3D& x) const override;

    //! Returns divergence at given position \p x.
    double divergence(const Vector3D& x) const override;

    //! Returns curl at given position \p x.
    Vector3D curl(const Vector3D& x) const override;

    //!
    //! \brief Returns the sampler function.
    //!
    //! This function returns the data sampler function object. The sampling
    //! function is linear.
    //!
    std::function<Vector3D(const Vector3D&)> sampler() const override;

    //! Returns builder fox FaceCenteredGrid3.
    static Builder builder();

 protected:
    // VectorGrid3 implementations
    void onResize(const Size3& resolution, const Vector3D& gridSpacing,
                  const Vector3D& origin, const Vector3D& initialValue) final;

    //! Fetches the data into a continuous linear array.
    void getData(std::vector<double>* data) const override;

    //! Sets the data from a continuous linear array.
    void setData(const std::vector<double>& data) override;

 private:
    Array3<double> _dataU;
    Array3<double> _dataV;
    Array3<double> _dataW;
    Vector3D _dataOriginU;
    Vector3D _dataOriginV;
    Vector3D _dataOriginW;
    LinearArraySampler3<double, double> _uLinearSampler;
    LinearArraySampler3<double, double> _vLinearSampler;
    LinearArraySampler3<double, double> _wLinearSampler;
    std::function<Vector3D(const Vector3D&)> _sampler;

    void resetSampler();
};

//! Shared pointer type for the FaceCenteredGrid3.
typedef std::shared_ptr<FaceCenteredGrid3> FaceCenteredGrid3Ptr;

//!
//! \brief Front-end to create CellCenteredScalarGrid3 objects step by step.
//!
class FaceCenteredGrid3::Builder final : public VectorGridBuilder3 {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Size3& resolution);

    //! Returns builder with resolution.
    Builder& withResolution(size_t resolutionX, size_t resolutionY,
                            size_t resolutionZ);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(const Vector3D& gridSpacing);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(double gridSpacingX, double gridSpacingY,
                             double gridSpacingZ);

    //! Returns builder with grid origin.
    Builder& withOrigin(const Vector3D& gridOrigin);

    //! Returns builder with grid origin.
    Builder& withOrigin(double gridOriginX, double gridOriginY,
                        double gridOriginZ);

    //! Returns builder with initial value.
    Builder& withInitialValue(const Vector3D& initialVal);

    //! Returns builder with initial value.
    Builder& withInitialValue(double initialValX, double initialValY,
                              double initialValZ);

    //! Builds CellCenteredScalarGrid3 instance.
    FaceCenteredGrid3 build() const;

    //! Builds shared pointer of FaceCenteredGrid3 instance.
    FaceCenteredGrid3Ptr makeShared() const;

    //!
    //! \brief Builds shared pointer of FaceCenteredGrid3 instance.
    //!
    //! This is an overriding function that implements VectorGridBuilder3.
    //!
    VectorGrid3Ptr build(const Size3& resolution, const Vector3D& gridSpacing,
                         const Vector3D& gridOrigin,
                         const Vector3D& initialVal) const override;

 private:
    Size3 _resolution{1, 1, 1};
    Vector3D _gridSpacing{1, 1, 1};
    Vector3D _gridOrigin{0, 0, 0};
    Vector3D _initialVal{0, 0, 0};
};

}  // namespace jet

#endif  // INCLUDE_JET_FACE_CENTERED_GRID3_H_
