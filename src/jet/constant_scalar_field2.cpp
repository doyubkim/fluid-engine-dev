// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

ConstantScalarField2Ptr ConstantScalarField2::Builder::makeShared() const {
    return std::shared_ptr<ConstantScalarField2>(
        new ConstantScalarField2(_value),
        [] (ConstantScalarField2* obj) {
            delete obj;
        });
}
