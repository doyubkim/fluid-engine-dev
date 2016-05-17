// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FACE_CENTERED_GRID3_H_
#define INCLUDE_JET_FACE_CENTERED_GRID3_H_

#include <jet/array3.h>
#include <jet/array_samplers3.h>
#include <jet/vector_grid3.h>
#include <memory>

namespace jet {

class FaceCenteredGrid3 final : public VectorGrid3 {
 public:
    typedef ArrayAccessor3<double> ScalarDataAccessor;
    typedef ConstArrayAccessor3<double> ConstScalarDataAccessor;

    FaceCenteredGrid3();

    explicit FaceCenteredGrid3(
        size_t resolutionX,
        size_t resolutionY,
        size_t resolutionZ,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double gridSpacingZ = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double originZ = 0.0,
        double initialValueU = 0.0,
        double initialValueV = 0.0,
        double initialValueW = 0.0);

    explicit FaceCenteredGrid3(
        const Size3& resolution,
        const Vector3D& gridSpacing = Vector3D(1.0, 1.0, 1.0),
        const Vector3D& origin = Vector3D(),
        const Vector3D& initialValue = Vector3D());

    FaceCenteredGrid3(const FaceCenteredGrid3& other);

    void swap(Grid3* other) override;

    void set(const FaceCenteredGrid3& other);

    FaceCenteredGrid3& operator=(const FaceCenteredGrid3& other);

    double& u(size_t i, size_t j, size_t k);

    const double& u(size_t i, size_t j, size_t k) const;

    double& v(size_t i, size_t j, size_t k);

    const double& v(size_t i, size_t j, size_t k) const;

    double& w(size_t i, size_t j, size_t k);

    const double& w(size_t i, size_t j, size_t k) const;

    Vector3D valueAtCellCenter(size_t i, size_t j, size_t k) const;

    //! Returns divergence at cell-center location.
    //! \param i Cell index i.
    //! \param j Cell index j.
    //! \param k Cell index k.
    double divergenceAtCellCenter(size_t i, size_t j, size_t k) const;

    //! Returns curl at cell-center location.
    //! \param i Cell index i.
    //! \param j Cell index j.
    //! \param k Cell index k.
    Vector3D curlAtCellCenter(size_t i, size_t j, size_t k) const;

    ScalarDataAccessor uAccessor();

    ConstScalarDataAccessor uConstAccessor() const;

    ScalarDataAccessor vAccessor();

    ConstScalarDataAccessor vConstAccessor() const;

    ScalarDataAccessor wAccessor();

    ConstScalarDataAccessor wConstAccessor() const;

    DataPositionFunc uPosition() const;

    DataPositionFunc vPosition() const;

    DataPositionFunc wPosition() const;

    Size3 uSize() const;

    Size3 vSize() const;

    Size3 wSize() const;

    //! Returns u-data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector3D uOrigin() const;

    //! Returns v-data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector3D vOrigin() const;

    //! Returns w-data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector3D wOrigin() const;

    void fill(const Vector3D& value) override;

    void fill(const std::function<Vector3D(const Vector3D&)>& func) override;

    std::shared_ptr<VectorGrid3> clone() const override;

    void forEachUIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    void parallelForEachUIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    void forEachVIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    void parallelForEachVIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    void forEachWIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    void parallelForEachWIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //! Serializes the grid instance to the output stream \p strm.
    void serialize(std::ostream* strm) const override;

    //! Deserializes the input stream \p strm to the grid instance.
    void deserialize(std::istream* strm) override;

    // VectorField3 implementations
    Vector3D sample(const Vector3D& x) const override;

    double divergence(const Vector3D& x) const override;

    Vector3D curl(const Vector3D& x) const override;

    std::function<Vector3D(const Vector3D&)> sampler() const override;

    static VectorGridBuilder3Ptr builder();

 protected:
    // VectorGrid3 implementations
    void onResize(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& origin,
        const Vector3D& initialValue) override;

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

typedef std::shared_ptr<FaceCenteredGrid3> FaceCenteredGrid3Ptr;


class FaceCenteredGridBuilder3 final : public VectorGridBuilder3 {
 public:
    FaceCenteredGridBuilder3();

    VectorGrid3Ptr build(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin,
        const Vector3D& initialVal) const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_FACE_CENTERED_GRID3_H_
