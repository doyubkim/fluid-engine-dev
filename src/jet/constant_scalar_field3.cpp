// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

ConstantScalarField3::Builder ConstantScalarField3::builder() {
    return Builder();
}


ConstantScalarField3::Builder&
ConstantScalarField3::Builder::withValue(double value) {
    _value = value;
    return *this;
}

ConstantScalarField3 ConstantScalarField3::Builder::build() const {
    return ConstantScalarField3(_value);
}

ConstantScalarField3Ptr ConstantScalarField3::Builder::makeShared() const {
    return std::shared_ptr<ConstantScalarField3>(
        new ConstantScalarField3(_value),
        [] (ConstantScalarField3* obj) {
            delete obj;
        });
}
