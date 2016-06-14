// Copyright (c) 2016 Doyub Kim

#include <jet/triangle_mesh3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(TriangleMesh3, Constructors) {
    TriangleMesh3 mesh1;
    EXPECT_EQ(0u, mesh1.numberOfPoints());
    EXPECT_EQ(0u, mesh1.numberOfNormals());
    EXPECT_EQ(0u, mesh1.numberOfUvs());
    EXPECT_EQ(0u, mesh1.numberOfTriangles());
}
