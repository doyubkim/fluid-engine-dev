// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FACE_CENTERED_GRID2_H_
#define INCLUDE_JET_FACE_CENTERED_GRID2_H_

#include <jet/array2.h>
#include <jet/array_samplers2.h>
#include <jet/vector_grid2.h>
#include <memory>

namespace jet {

class FaceCenteredGrid2 final : public VectorGrid2 {
 public:
    typedef ArrayAccessor2<double> ScalarDataAccessor;
    typedef ConstArrayAccessor2<double> ConstScalarDataAccessor;

    FaceCenteredGrid2();

    explicit FaceCenteredGrid2(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double initialValueU = 0.0,
        double initialValueV = 0.0);

    explicit FaceCenteredGrid2(
        const Size2& resolution,
        const Vector2D& gridSpacing = Vector2D(1.0, 1.0),
        const Vector2D& origin = Vector2D(),
        const Vector2D& initialValue = Vector2D());

    FaceCenteredGrid2(const FaceCenteredGrid2& other);

    void swap(Grid2* other) override;

    void set(const FaceCenteredGrid2& other);

    FaceCenteredGrid2& operator=(const FaceCenteredGrid2& other);

    double& u(size_t i, size_t j);

    const double& u(size_t i, size_t j) const;

    double& v(size_t i, size_t j);

    const double& v(size_t i, size_t j) const;

    Vector2D valueAtCellCenter(size_t i, size_t j) const;

    //! Returns divergence at cell-center location.
    //! \param i Cell index i.
    //! \param j Cell index j.
    double divergenceAtCellCenter(size_t i, size_t j) const;

    //! Returns curl at cell-center location.
    //! \param i Cell index i.
    //! \param j Cell index j.
    double curlAtCellCenter(size_t i, size_t j) const;

    ScalarDataAccessor uAccessor();

    ConstScalarDataAccessor uConstAccessor() const;

    ScalarDataAccessor vAccessor();

    ConstScalarDataAccessor vConstAccessor() const;

    DataPositionFunc uPosition() const;

    DataPositionFunc vPosition() const;

    Size2 uSize() const;

    Size2 vSize() const;

    //! Returns u-data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector2D uOrigin() const;

    //! Returns v-data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector2D vOrigin() const;

    void fill(const Vector2D& value) override;

    void fill(const std::function<Vector2D(const Vector2D&)>& func) override;

    std::shared_ptr<VectorGrid2> clone() const override;

    void forEachUIndex(
        const std::function<void(size_t, size_t)>& func) const;

    void parallelForEachUIndex(
        const std::function<void(size_t, size_t)>& func) const;

    void forEachVIndex(
        const std::function<void(size_t, size_t)>& func) const;

    void parallelForEachVIndex(
        const std::function<void(size_t, size_t)>& func) const;

    void forEachWIndex(
        const std::function<void(size_t, size_t)>& func) const;

    void parallelForEachWIndex(
        const std::function<void(size_t, size_t)>& func) const;

    //! Serializes the grid instance to the output stream \p strm.
    void serialize(std::ostream* strm) const override;

    //! Deserializes the input stream \p strm to the grid instance.
    void deserialize(std::istream* strm) override;

    // VectorField2 implementations
    Vector2D sample(const Vector2D& x) const override;

    double divergence(const Vector2D& x) const override;

    double curl(const Vector2D& x) const override;

    std::function<Vector2D(const Vector2D&)> sampler() const override;

    static VectorGridBuilder2Ptr builder();

 protected:
    // VectorGrid2 implementations
    void onResize(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& origin,
        const Vector2D& initialValue) override;

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

typedef std::shared_ptr<FaceCenteredGrid2> FaceCenteredGrid2Ptr;


class FaceCenteredGridBuilder2 final : public VectorGridBuilder2 {
 public:
    FaceCenteredGridBuilder2();

    VectorGrid2Ptr build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        const Vector2D& initialVal) const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_FACE_CENTERED_GRID2_H_
