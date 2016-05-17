// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SCALAR_FIELD2_H_
#define INCLUDE_JET_SCALAR_FIELD2_H_

#include <jet/field2.h>
#include <jet/vector2.h>
#include <functional>
#include <memory>

namespace jet {

class ScalarField2 : public Field2 {
 public:
    ScalarField2();

    virtual ~ScalarField2();

    virtual double sample(const Vector2D& x) const = 0;

    virtual Vector2D gradient(const Vector2D& x) const;

    virtual double laplacian(const Vector2D& x) const;

    virtual std::function<double(const Vector2D&)> sampler() const;
};

typedef std::shared_ptr<ScalarField2> ScalarField2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_FIELD2_H_
