// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/constant_vector_field3.h>

using namespace jet;

ConstantVectorField3::ConstantVectorField3(const Vector3D& value) :
    _value(value) {
}

Vector3D ConstantVectorField3::sample(const Vector3D& x) const {
    UNUSED_VARIABLE(x);

    return _value;
}

std::function<Vector3D(const Vector3D&)> ConstantVectorField3::sampler() const {
    return [this](const Vector3D&) -> Vector3D {
        return _value;
    };
}

ConstantVectorField3::Builder ConstantVectorField3::builder() {
    return Builder();
}


ConstantVectorField3::Builder&
ConstantVectorField3::Builder::withValue(const Vector3D& value) {
    _value = value;
    return *this;
}

ConstantVectorField3 ConstantVectorField3::Builder::build() const {
    return ConstantVectorField3(_value);
}

ConstantVectorField3Ptr ConstantVectorField3::Builder::makeShared() const {
    return std::shared_ptr<ConstantVectorField3>(
        new ConstantVectorField3(_value),
        [] (ConstantVectorField3* obj) {
            delete obj;
        });
}
