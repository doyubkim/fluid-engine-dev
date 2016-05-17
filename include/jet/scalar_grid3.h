// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SCALAR_GRID3_H_
#define INCLUDE_JET_SCALAR_GRID3_H_

#include <jet/array3.h>
#include <jet/array_accessor3.h>
#include <jet/array_samplers3.h>
#include <jet/grid3.h>
#include <jet/scalar_field3.h>
#include <memory>

namespace jet {

class ScalarGrid3 : public ScalarField3, public Grid3 {
 public:
    typedef ArrayAccessor3<double> ScalarDataAccessor;
    typedef ConstArrayAccessor3<double> ConstScalarDataAccessor;

    ScalarGrid3();

    virtual ~ScalarGrid3();

    virtual Size3 dataSize() const = 0;

    //! Returns data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    virtual Vector3D dataOrigin() const = 0;

    virtual std::shared_ptr<ScalarGrid3> clone() const = 0;

    void clear();

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

    void resize(
        const Size3& resolution,
        const Vector3D& gridSpacing = Vector3D(1, 1, 1),
        const Vector3D& origin = Vector3D(),
        double initialValue = 0.0);

    void resize(
        double gridSpacingX,
        double gridSpacingY,
        double gridSpacingZ,
        double originX,
        double originY,
        double originZ);

    void resize(const Vector3D& gridSpacing, const Vector3D& origin);

    const double& operator()(size_t i, size_t j, size_t k) const;

    double& operator()(size_t i, size_t j, size_t k);

    Vector3D gradientAtDataPoint(size_t i, size_t j, size_t k) const;

    double laplacianAtDataPoint(size_t i, size_t j, size_t k) const;

    ScalarDataAccessor dataAccessor();

    ConstScalarDataAccessor constDataAccessor() const;

    DataPositionFunc dataPosition() const;

    void fill(double value);

    void fill(const std::function<double(const Vector3D&)>& func);

    void forEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    void parallelForEachDataPointIndex(
        const std::function<void(size_t, size_t, size_t)>& func) const;

    //! Serializes the grid instance to the output stream \p strm.
    void serialize(std::ostream* strm) const override;

    //! Deserializes the input stream \p strm to the grid instance.
    void deserialize(std::istream* strm) override;

    // ScalarField3 implementations
    double sample(const Vector3D& x) const override;

    std::function<double(const Vector3D&)> sampler() const override;

    Vector3D gradient(const Vector3D& x) const override;

    double laplacian(const Vector3D& x) const override;

 protected:
    void swapScalarGrid(ScalarGrid3* other);

    void setScalarGrid(const ScalarGrid3& other);

 private:
    Array3<double> _data;
    LinearArraySampler3<double, double> _linearSampler;
    std::function<double(const Vector3D&)> _sampler;

    void resetSampler();
};

typedef std::shared_ptr<ScalarGrid3> ScalarGrid3Ptr;


class ScalarGridBuilder3 {
 public:
    ScalarGridBuilder3();

    virtual ~ScalarGridBuilder3();

    virtual ScalarGrid3Ptr build(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin,
        double initialVal) const = 0;
};

typedef std::shared_ptr<ScalarGridBuilder3> ScalarGridBuilder3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_GRID3_H_
