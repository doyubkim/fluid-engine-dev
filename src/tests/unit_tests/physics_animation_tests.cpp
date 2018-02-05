// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/physics_animation.h>

#include <gtest/gtest.h>

using namespace jet;

class CustomPhysicsAnimation : public PhysicsAnimation {
 public:
 protected:
    void onAdvanceTimeStep(double timeIntervalInSeconds) override {
        (void)timeIntervalInSeconds;
    }
};

TEST(PhysicsAnimation, Constructors) {
    CustomPhysicsAnimation pa;
    EXPECT_EQ(-1, pa.currentFrame().index);
}

TEST(PhysicsAnimation, Properties) {
    CustomPhysicsAnimation pa;

    pa.setIsUsingFixedSubTimeSteps(true);
    EXPECT_TRUE(pa.isUsingFixedSubTimeSteps());
    pa.setIsUsingFixedSubTimeSteps(false);
    EXPECT_FALSE(pa.isUsingFixedSubTimeSteps());

    pa.setNumberOfFixedSubTimeSteps(42);
    EXPECT_EQ(42u, pa.numberOfFixedSubTimeSteps());

    pa.setCurrentFrame(Frame(8, 0.01));
    EXPECT_EQ(8, pa.currentFrame().index);
    EXPECT_EQ(0.01, pa.currentFrame().timeIntervalInSeconds);
}

TEST(PhysicsAnimation, Updates) {
    CustomPhysicsAnimation pa;

    for (Frame frame(0, 0.1); frame.index <= 15; ++frame) {
        pa.update(frame);
    }

    EXPECT_DOUBLE_EQ(1.5, pa.currentTimeInSeconds());

    CustomPhysicsAnimation pa2;

    for (int i = 0; i <= 15; ++i) {
        pa2.advanceSingleFrame();
    }

    EXPECT_DOUBLE_EQ(pa2.currentFrame().timeIntervalInSeconds * 15.0,
                     pa2.currentTimeInSeconds());
}