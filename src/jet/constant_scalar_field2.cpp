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

ConstantScalarField2::Builder ConstantScalarField2::builder() {
    return Builder();
}


ConstantScalarField2::Builder&
ConstantScalarField2::Builder::withValue(double value) {
    _value = value;
    return *this;
}

ConstantScalarField2 ConstantScalarField2::Builder::build() const {
    return ConstantScalarField2(_value);
}
