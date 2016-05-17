// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/custom_scalar_field2.h>

using namespace jet;

CustomScalarField2::CustomScalarField2(
    const std::function<double(const Vector2D&)>& customFunction,
    double derivativeResolution) :
    _customFunction(customFunction),
    _resolution(derivativeResolution) {
}

CustomScalarField2::CustomScalarField2(
    const std::function<double(const Vector2D&)>& customFunction,
    const std::function<Vector2D(const Vector2D&)>& customGradientFunction,
    double derivativeResolution) :
    _customFunction(customFunction),
    _customGradientFunction(customGradientFunction),
    _resolution(derivativeResolution) {
}

CustomScalarField2::CustomScalarField2(
    const std::function<double(const Vector2D&)>& customFunction,
    const std::function<Vector2D(const Vector2D&)>& customGradientFunction,
    const std::function<double(const Vector2D&)>& customLaplacianFunction) :
    _customFunction(customFunction),
    _customGradientFunction(customGradientFunction),
    _customLaplacianFunction(customLaplacianFunction),
    _resolution(1e-3) {
}

double CustomScalarField2::sample(const Vector2D& x) const {
    return _customFunction(x);
}

std::function<double(const Vector2D&)> CustomScalarField2::sampler() const {
    return _customFunction;
}

Vector2D CustomScalarField2::gradient(const Vector2D& x) const {
    if (_customGradientFunction) {
        return _customGradientFunction(x);
    } else {
        double left = _customFunction(x - Vector2D(0.5 * _resolution, 0.0));
        double right = _customFunction(x + Vector2D(0.5 * _resolution, 0.0));
        double bottom = _customFunction(x - Vector2D(0.0, 0.5 * _resolution));
        double top = _customFunction(x + Vector2D(0.0, 0.5 * _resolution));

        return Vector2D(
            (right - left) / _resolution,
            (top - bottom) / _resolution);
    }
}

double CustomScalarField2::laplacian(const Vector2D& x) const {
    if (_customLaplacianFunction) {
        return _customLaplacianFunction(x);
    } else {
        double center = _customFunction(x);
        double left = _customFunction(x - Vector2D(0.5 * _resolution, 0.0));
        double right = _customFunction(x + Vector2D(0.5 * _resolution, 0.0));
        double bottom = _customFunction(x - Vector2D(0.0, 0.5 * _resolution));
        double top = _customFunction(x + Vector2D(0.0, 0.5 * _resolution));

        return (left + right + bottom + top - 4.0 * center)
            / (_resolution * _resolution);
    }
}
