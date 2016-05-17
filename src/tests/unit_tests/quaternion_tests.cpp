// Copyright (c) 2016 Doyub Kim

#include <jet/quaternion.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Quaternion, Constructors) {
}

TEST(Quaternion, BasicSetters) {
    {
        QuaternionD q;
        q.set(QuaternionD(1, 2, 3, 4));

        EXPECT_DOUBLE_EQ(1.0, q.w);
        EXPECT_DOUBLE_EQ(2.0, q.x);
        EXPECT_DOUBLE_EQ(3.0, q.y);
        EXPECT_DOUBLE_EQ(4.0, q.z);
    }
    {
        QuaternionD q;
        q.set(1, 2, 3, 4);

        EXPECT_DOUBLE_EQ(1.0, q.w);
        EXPECT_DOUBLE_EQ(2.0, q.x);
        EXPECT_DOUBLE_EQ(3.0, q.y);
        EXPECT_DOUBLE_EQ(4.0, q.z);
    }
    {
        QuaternionD q;
        q.set({ 1, 2, 3, 4 });

        EXPECT_DOUBLE_EQ(1.0, q.w);
        EXPECT_DOUBLE_EQ(2.0, q.x);
        EXPECT_DOUBLE_EQ(3.0, q.y);
        EXPECT_DOUBLE_EQ(4.0, q.z);
    }
    {
        // set with axis & angle
        Vector3D originalAxis = Vector3D(1, 3, 2).normalized();
        double originalAngle = 0.4;

        QuaternionD q;
        q.set(originalAxis, originalAngle);

        Vector3D axis = q.axis();
        double angle = q.angle();

        EXPECT_DOUBLE_EQ(originalAxis.x, axis.x);
        EXPECT_DOUBLE_EQ(originalAxis.y, axis.y);
        EXPECT_DOUBLE_EQ(originalAxis.z, axis.z);
        EXPECT_DOUBLE_EQ(originalAngle, angle);
    }

    {
        // set with from & to vectors (90 degrees)
        Vector3D from(1, 0, 0);
        Vector3D to(0, 0, 1);

        QuaternionD q;
        q.set(from, to);

        Vector3D axis = q.axis();
        double angle = q.angle();

        EXPECT_DOUBLE_EQ(0.0, axis.x);
        EXPECT_DOUBLE_EQ(-1.0, axis.y);
        EXPECT_DOUBLE_EQ(0.0, axis.z);
        EXPECT_DOUBLE_EQ(pi<double>()/2.0, angle);
    }
    {
        Vector3D rotationBasis0(1, 0, 0);
        Vector3D rotationBasis1(0, 0, 1);
        Vector3D rotationBasis2(0, -1, 0);

        QuaternionD q;
        q.set(rotationBasis0, rotationBasis1, rotationBasis2);

        EXPECT_DOUBLE_EQ(std::sqrt(2.0) / 2.0, q.w);
        EXPECT_DOUBLE_EQ(std::sqrt(2.0) / 2.0, q.x);
        EXPECT_DOUBLE_EQ(0.0, q.y);
        EXPECT_DOUBLE_EQ(0.0, q.z);
    }
}
