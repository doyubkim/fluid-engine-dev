// Copyright (c) 2016 Doyub Kim

#include <jet/cylinder3.h>
#include <gtest/gtest.h>
#include <limits>

using namespace jet;

TEST(Cylinder3, Constructors) {
    Cylinder3 cyl1;
    EXPECT_DOUBLE_EQ(0.0, cyl1.center().x);
    EXPECT_DOUBLE_EQ(0.0, cyl1.center().y);
    EXPECT_DOUBLE_EQ(0.0, cyl1.center().z);
    EXPECT_DOUBLE_EQ(1.0, cyl1.radius());
    EXPECT_DOUBLE_EQ(1.0, cyl1.height());
    EXPECT_DOUBLE_EQ(-1.0, cyl1.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(-0.5, cyl1.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, cyl1.boundingBox().lowerCorner.z);
    EXPECT_DOUBLE_EQ(1.0, cyl1.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(0.5, cyl1.boundingBox().upperCorner.y);
    EXPECT_DOUBLE_EQ(1.0, cyl1.boundingBox().upperCorner.z);

    Cylinder3 cyl2(Vector3D(1, 2, 3), 4.0, 5.0);
    EXPECT_DOUBLE_EQ(1.0, cyl2.center().x);
    EXPECT_DOUBLE_EQ(2.0, cyl2.center().y);
    EXPECT_DOUBLE_EQ(3.0, cyl2.center().z);
    EXPECT_DOUBLE_EQ(4.0, cyl2.radius());
    EXPECT_DOUBLE_EQ(5.0, cyl2.height());
    EXPECT_DOUBLE_EQ(-3.0, cyl2.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(-0.5, cyl2.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, cyl2.boundingBox().lowerCorner.z);
    EXPECT_DOUBLE_EQ(5.0, cyl2.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(4.5, cyl2.boundingBox().upperCorner.y);
    EXPECT_DOUBLE_EQ(7.0, cyl2.boundingBox().upperCorner.z);

    Cylinder3 cyl3(cyl2);
    EXPECT_DOUBLE_EQ(1.0, cyl3.center().x);
    EXPECT_DOUBLE_EQ(2.0, cyl3.center().y);
    EXPECT_DOUBLE_EQ(3.0, cyl3.center().z);
    EXPECT_DOUBLE_EQ(4.0, cyl3.radius());
    EXPECT_DOUBLE_EQ(5.0, cyl3.height());
    EXPECT_DOUBLE_EQ(-3.0, cyl3.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(-0.5, cyl3.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, cyl3.boundingBox().lowerCorner.z);
    EXPECT_DOUBLE_EQ(5.0, cyl3.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(4.5, cyl3.boundingBox().upperCorner.y);
    EXPECT_DOUBLE_EQ(7.0, cyl3.boundingBox().upperCorner.z);
}
