// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/box3.h>
#include <jet/collider_set3.h>
#include <jet/rigid_body_collider3.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(ColliderSet3, Constructors) {
    auto box1 = Box3::builder()
        .withLowerCorner({0, 1, 2})
        .withUpperCorner({1, 2, 3})
        .makeShared();

    auto box2 = Box3::builder()
        .withLowerCorner({3, 4, 5})
        .withUpperCorner({4, 5, 6})
        .makeShared();

    auto col1 = RigidBodyCollider3::builder()
        .withSurface(box1)
        .makeShared();

    auto col2 = RigidBodyCollider3::builder()
        .withSurface(box2)
        .makeShared();

    ColliderSet3 colSet1;
    EXPECT_EQ(0u, colSet1.numberOfColliders());

    ColliderSet3 colSet3({col1, col2});
    EXPECT_EQ(2u, colSet3.numberOfColliders());
    EXPECT_EQ(col1, colSet3.collider(0));
    EXPECT_EQ(col2, colSet3.collider(1));
}

TEST(ColliderSet3, Builder) {
    auto box1 = Box3::builder()
        .withLowerCorner({0, 1, 2})
        .withUpperCorner({1, 2, 3})
        .makeShared();

    auto box2 = Box3::builder()
        .withLowerCorner({3, 4, 5})
        .withUpperCorner({4, 5, 6})
        .makeShared();

    auto col1 = RigidBodyCollider3::builder()
        .withSurface(box1)
        .makeShared();

    auto col2 = RigidBodyCollider3::builder()
        .withSurface(box2)
        .makeShared();

    auto colSet1 = ColliderSet3::builder().makeShared();
    EXPECT_EQ(0u, colSet1->numberOfColliders());

    auto colSet2 = ColliderSet3::builder()
        .withColliders({col1, col2})
        .makeShared();
    EXPECT_EQ(2u, colSet2->numberOfColliders());
    EXPECT_EQ(col1, colSet2->collider(0));
    EXPECT_EQ(col2, colSet2->collider(1));

    auto colSet3 = ColliderSet3::builder().build();
    EXPECT_EQ(0u, colSet3.numberOfColliders());
}
