// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VECTOR_FIELD3_H_
#define INCLUDE_JET_VECTOR_FIELD3_H_

#include <jet/field3.h>
#include <jet/vector3.h>
#include <functional>
#include <memory>

namespace jet {

class VectorField3 : public Field3 {
 public:
    VectorField3();

    virtual ~VectorField3();

    virtual Vector3D sample(const Vector3D& x) const = 0;

    virtual double divergence(const Vector3D& x) const;

    virtual Vector3D curl(const Vector3D& x) const;

    virtual std::function<Vector3D(const Vector3D&)> sampler() const;
};

typedef std::shared_ptr<VectorField3> VectorField3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_VECTOR_FIELD3_H_
