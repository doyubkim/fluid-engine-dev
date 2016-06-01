// Copyright (c) 2016 Doyub Kim

#include <jet/box2.h>
#include <jet/implicit_surface_set2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(ImplicitSurfaceSet2, ClosestNormal) {
    BoundingBox2D bbox(Vector2D(), Vector2D(1, 2));

    Box2Ptr box = std::make_shared<Box2>(bbox);
    box->setIsNormalFlipped(true);

    ImplicitSurfaceSet2Ptr sset = std::make_shared<ImplicitSurfaceSet2>();
    sset->addSurface(box);

    Vector2D pt(0.5, 2.5);
    Vector2D boxNormal = box->closestNormal(pt);
    Vector2D setNormal = sset->closestNormal(pt);
    EXPECT_DOUBLE_EQ(boxNormal.x, setNormal.x);
    EXPECT_DOUBLE_EQ(boxNormal.y, setNormal.y);
}
