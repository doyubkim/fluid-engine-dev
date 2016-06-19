// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_
#define INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_

#include <jet/vector_field3.h>

namespace jet {

//! 3-D vector field with custom field function.
class CustomVectorField3 final : public VectorField3 {
 public:
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

 private:
    std::function<Vector3D(const Vector3D&)> _customFunction;
    std::function<double(const Vector3D&)> _customDivergenceFunction;
    std::function<Vector3D(const Vector3D&)> _customCurlFunction;
    double _resolution = 1e-3;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_
