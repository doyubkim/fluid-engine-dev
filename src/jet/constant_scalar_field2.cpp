// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constant_scalar_field2.h>

using namespace jet;

ConstantScalarField2::ConstantScalarField2(double value) :
    _value(value) {
}

double ConstantScalarField2::sample(const Vector2D& x) const {
    UNUSED_VARIABLE(x);

    return _value;
}

std::function<double(const Vector2D&)> ConstantScalarField2::sampler() const {
    double value = _value;
    return [value](const Vector2D&) -> double {
        return value;
    };
}
