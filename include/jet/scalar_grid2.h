// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SCALAR_GRID2_H_
#define INCLUDE_JET_SCALAR_GRID2_H_

#include <jet/array2.h>
#include <jet/array_accessor2.h>
#include <jet/array_samplers2.h>
#include <jet/grid2.h>
#include <jet/scalar_field2.h>
#include <memory>

namespace jet {

class ScalarGrid2 : public ScalarField2, public Grid2 {
 public:
    typedef ArrayAccessor2<double> ScalarDataAccessor;
    typedef ConstArrayAccessor2<double> ConstScalarDataAccessor;

    ScalarGrid2();

    virtual ~ScalarGrid2();

    virtual Size2 dataSize() const = 0;

    //! Returns data position for the grid point at (0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    virtual Vector2D dataOrigin() const = 0;

    virtual std::shared_ptr<ScalarGrid2> clone() const = 0;

    void clear();

    void resize(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double initialValue = 0.0);

    void resize(
        const Size2& resolution,
        const Vector2D& gridSpacing = Vector2D(1, 1),
        const Vector2D& origin = Vector2D(),
        double initialValue = 0.0);

    void resize(
        double gridSpacingX,
        double gridSpacingY,
        double originX,
        double originY);

    void resize(const Vector2D& gridSpacing, const Vector2D& origin);

    const double& operator()(size_t i, size_t j) const;

    double& operator()(size_t i, size_t j);

    Vector2D gradientAtDataPoint(size_t i, size_t j) const;

    double laplacianAtDataPoint(size_t i, size_t j) const;

    ScalarDataAccessor dataAccessor();

    ConstScalarDataAccessor constDataAccessor() const;

    DataPositionFunc dataPosition() const;

    void fill(double value);

    void fill(const std::function<double(const Vector2D&)>& func);

    void forEachDataPointIndex(
        const std::function<void(size_t, size_t)>& func) const;

    void parallelForEachDataPointIndex(
        const std::function<void(size_t, size_t)>& func) const;

    //! Serializes the grid instance to the output stream \p strm.
    void serialize(std::ostream* strm) const override;

    //! Deserializes the input stream \p strm to the grid instance.
    void deserialize(std::istream* strm) override;

    // ScalarField2 implementations
    double sample(const Vector2D& x) const override;

    std::function<double(const Vector2D&)> sampler() const override;

    Vector2D gradient(const Vector2D& x) const override;

    double laplacian(const Vector2D& x) const override;

 protected:
    void swapScalarGrid(ScalarGrid2* other);

    void setScalarGrid(const ScalarGrid2& other);

 private:
    Array2<double> _data;
    LinearArraySampler2<double, double> _linearSampler;
    std::function<double(const Vector2D&)> _sampler;

    void resetSampler();
};

typedef std::shared_ptr<ScalarGrid2> ScalarGrid2Ptr;


class ScalarGridBuilder2 {
 public:
    ScalarGridBuilder2();

    virtual ~ScalarGridBuilder2();

    virtual ScalarGrid2Ptr build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        double initialVal) const = 0;
};

typedef std::shared_ptr<ScalarGridBuilder2> ScalarGridBuilder2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_GRID2_H_
