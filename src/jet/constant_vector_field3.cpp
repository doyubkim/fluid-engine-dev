// Copyright (c) 2016 Doyub Kim

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
