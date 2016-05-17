// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_COLLIDER_SET3_H_
#define INCLUDE_JET_COLLIDER_SET3_H_

#include <jet/collider3.h>
#include <jet/surface_set3.h>
#include <vector>

namespace jet {

class ColliderSet3 final : public Collider3 {
 public:
    ColliderSet3();

    Vector3D velocityAt(const Vector3D& point) const override;

    void addCollider(const Collider3Ptr& collider);

 private:
    std::vector<Collider3Ptr> _colliders;
};

typedef std::shared_ptr<ColliderSet3> ColliderSet3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER_SET3_H_
