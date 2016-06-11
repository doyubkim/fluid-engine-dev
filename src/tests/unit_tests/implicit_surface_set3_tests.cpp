// Copyright (c) 2016 Doyub Kim

#include <jet/box3.h>
#include <jet/implicit_surface_set3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(ImplicitSurfaceSet3, ClosestNormal) {
    BoundingBox3D bbox(Vector3D(), Vector3D(1, 2, 3));

    Box3Ptr box = std::make_shared<Box3>(bbox);
    box->isNormalFlipped = true;

    ImplicitSurfaceSet3Ptr sset = std::make_shared<ImplicitSurfaceSet3>();
    sset->addSurface(box);

    Vector3D pt(0.5, 2.5, 2.0);
    Vector3D boxNormal = box->closestNormal(pt);
    Vector3D setNormal = sset->closestNormal(pt);
    EXPECT_DOUBLE_EQ(boxNormal.x, setNormal.x);
    EXPECT_DOUBLE_EQ(boxNormal.y, setNormal.y);
    EXPECT_DOUBLE_EQ(boxNormal.z, setNormal.z);
}
