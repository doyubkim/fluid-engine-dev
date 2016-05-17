// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_COLLIDER_SET2_H_
#define INCLUDE_JET_COLLIDER_SET2_H_

#include <jet/collider2.h>
#include <jet/surface_set2.h>
#include <vector>

namespace jet {

class ColliderSet2 final : public Collider2 {
 public:
    ColliderSet2();

    Vector2D velocityAt(const Vector2D& point) const override;

    void addCollider(const Collider2Ptr& collider);

 private:
    std::vector<Collider2Ptr> _colliders;
};

typedef std::shared_ptr<ColliderSet2> ColliderSet2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER_SET2_H_
