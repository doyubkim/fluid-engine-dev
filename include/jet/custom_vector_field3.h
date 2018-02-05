// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_
#define INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_

#include <jet/vector_field3.h>

namespace jet {

//! 3-D vector field with custom field function.
class CustomVectorField3 final : public VectorField3 {
 public:
    class Builder;

    //!
    //! \brief Constructs a field with given function.
    //!
    //! This constructor creates a field with user-provided function object.
    //! To compute derivatives, such as gradient and Laplacian, finite
    //! differencing is used. Thus, the differencing resolution also can be
    //! provided as the last parameter.
    //!
    CustomVectorField3(
        const std::function<Vector3D(const Vector3D&)>& customFunction,
        double derivativeResolution = 1e-3);

    //!
    //! \brief Constructs a field with given field and gradient function.
    //!
    //! This constructor creates a field with user-provided field and gradient
    //! function objects. To compute Laplacian, finite differencing is used.
    //! Thus, the differencing resolution also can be provided as the last
    //! parameter.
    //!
    CustomVectorField3(
        const std::function<Vector3D(const Vector3D&)>& customFunction,
        const std::function<double(const Vector3D&)>& customDivergenceFunction,
        double derivativeResolution = 1e-3);

    //! Constructs a field with given field, gradient, and Laplacian function.
    CustomVectorField3(
        const std::function<Vector3D(const Vector3D&)>& customFunction,
        const std::function<double(const Vector3D&)>& customDivergenceFunction,
        const std::function<Vector3D(const Vector3D&)>& customCurlFunction);

    //! Returns the sampled value at given position \p x.
    Vector3D sample(const Vector3D& x) const override;

    //! Returns the divergence at given position \p x.
    double divergence(const Vector3D& x) const override;

    //! Returns the curl at given position \p x.
    Vector3D curl(const Vector3D& x) const override;

    //! Returns the sampler function.
    std::function<Vector3D(const Vector3D&)> sampler() const override;

    //! Returns builder fox CustomVectorField2.
    static Builder builder();

 private:
    std::function<Vector3D(const Vector3D&)> _customFunction;
    std::function<double(const Vector3D&)> _customDivergenceFunction;
    std::function<Vector3D(const Vector3D&)> _customCurlFunction;
    double _resolution = 1e-3;
};

//! Shared pointer type for the CustomVectorField3.
typedef std::shared_ptr<CustomVectorField3> CustomVectorField3Ptr;


//!
//! \brief Front-end to create CustomVectorField3 objects step by step.
//!
class CustomVectorField3::Builder final {
 public:
    //! Returns builder with field function.
    Builder& withFunction(
        const std::function<Vector3D(const Vector3D&)>& func);

    //! Returns builder with divergence function.
    Builder& withDivergenceFunction(
        const std::function<double(const Vector3D&)>& func);

    //! Returns builder with curl function.
    Builder& withCurlFunction(
        const std::function<Vector3D(const Vector3D&)>& func);

    //! Returns builder with derivative resolution.
    Builder& withDerivativeResolution(double resolution);

    //! Builds CustomVectorField3.
    CustomVectorField3 build() const;

    //! Builds shared pointer of CustomVectorField3 instance.
    CustomVectorField3Ptr makeShared() const;

 private:
    double _resolution = 1e-3;
    std::function<Vector3D(const Vector3D&)> _customFunction;
    std::function<double(const Vector3D&)> _customDivergenceFunction;
    std::function<Vector3D(const Vector3D&)> _customCurlFunction;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_
