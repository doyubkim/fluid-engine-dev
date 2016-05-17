// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constant_scalar_field3.h>

using namespace jet;

ConstantScalarField3::ConstantScalarField3(double value) :
    _value(value) {
}

double ConstantScalarField3::sample(const Vector3D& x) const {
    UNUSED_VARIABLE(x);

    return _value;
}

std::function<double(const Vector3D&)> ConstantScalarField3::sampler() const {
    double value = _value;
    return [value](const Vector3D&) -> double {
        return value;
    };
}
