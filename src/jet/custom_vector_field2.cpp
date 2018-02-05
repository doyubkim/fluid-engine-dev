// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/custom_vector_field2.h>

using namespace jet;

CustomVectorField2::CustomVectorField2(
    const std::function<Vector2D(const Vector2D&)>& customFunction,
    double derivativeResolution) :
    _customFunction(customFunction),
    _resolution(derivativeResolution) {
}

CustomVectorField2::CustomVectorField2(
    const std::function<Vector2D(const Vector2D&)>& customFunction,
    const std::function<double(const Vector2D&)>& customDivergenceFunction,
    double derivativeResolution) :
    _customFunction(customFunction),
    _customDivergenceFunction(customDivergenceFunction),
    _resolution(derivativeResolution) {
}

CustomVectorField2::CustomVectorField2(
    const std::function<Vector2D(const Vector2D&)>& customFunction,
    const std::function<double(const Vector2D&)>& customDivergenceFunction,
    const std::function<double(const Vector2D&)>& customCurlFunction) :
    _customFunction(customFunction),
    _customDivergenceFunction(customDivergenceFunction),
    _customCurlFunction(customCurlFunction) {
}

Vector2D CustomVectorField2::sample(const Vector2D& x) const {
    return _customFunction(x);
}

std::function<Vector2D(const Vector2D&)> CustomVectorField2::sampler() const {
    return _customFunction;
}

double CustomVectorField2::divergence(const Vector2D& x) const {
    if (_customDivergenceFunction) {
        return _customDivergenceFunction(x);
    } else {
        double left = _customFunction(x - Vector2D(0.5 * _resolution, 0.0)).x;
        double right = _customFunction(x + Vector2D(0.5 * _resolution, 0.0)).x;
        double bottom = _customFunction(x - Vector2D(0.0, 0.5 * _resolution)).y;
        double top = _customFunction(x + Vector2D(0.0, 0.5 * _resolution)).y;

        return (right - left + top - bottom) / _resolution;
    }
}

double CustomVectorField2::curl(const Vector2D& x) const {
    if (_customCurlFunction) {
        return _customCurlFunction(x);
    } else {
        double left = _customFunction(x - Vector2D(0.5 * _resolution, 0.0)).y;
        double right = _customFunction(x + Vector2D(0.5 * _resolution, 0.0)).y;
        double bottom = _customFunction(x - Vector2D(0.0, 0.5 * _resolution)).x;
        double top = _customFunction(x + Vector2D(0.0, 0.5 * _resolution)).x;

        return (top - bottom - right + left) / _resolution;
    }
}

CustomVectorField2::Builder CustomVectorField2::builder() {
    return Builder();
}


CustomVectorField2::Builder&
CustomVectorField2::Builder::withFunction(
    const std::function<Vector2D(const Vector2D&)>& func) {
    _customFunction = func;
    return *this;
}

CustomVectorField2::Builder&
CustomVectorField2::Builder::withDivergenceFunction(
    const std::function<double(const Vector2D&)>& func) {
    _customDivergenceFunction = func;
    return *this;
}

CustomVectorField2::Builder&
CustomVectorField2::Builder::withCurlFunction(
    const std::function<double(const Vector2D&)>& func) {
    _customCurlFunction = func;
    return *this;
}

CustomVectorField2::Builder&
CustomVectorField2::Builder::withDerivativeResolution(double resolution) {
    _resolution = resolution;
    return *this;
}

CustomVectorField2 CustomVectorField2::Builder::build() const {
    if (_customCurlFunction) {
        return CustomVectorField2(
            _customFunction,
            _customDivergenceFunction,
            _customCurlFunction);
    } else {
        return CustomVectorField2(
            _customFunction,
            _customDivergenceFunction,
            _resolution);
    }
}

CustomVectorField2Ptr CustomVectorField2::Builder::makeShared() const {
    if (_customCurlFunction) {
        return std::shared_ptr<CustomVectorField2>(
            new CustomVectorField2(
                _customFunction,
                _customDivergenceFunction,
                _customCurlFunction),
            [] (CustomVectorField2* obj) {
                delete obj;
            });
    } else {
        return std::shared_ptr<CustomVectorField2>(
            new CustomVectorField2(
                _customFunction,
                _customDivergenceFunction,
                _resolution),
            [] (CustomVectorField2* obj) {
                delete obj;
            });
    }
}
