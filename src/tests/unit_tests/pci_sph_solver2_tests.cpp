// Copyright (c) 2016 Doyub Kim

#include <jet/pci_sph_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PciSphSolver2, UpdateEmpty) {
    // Empty solver test
    PciSphSolver2 solver;
    Frame frame(1, 0.01);
    solver.update(frame);
    solver.update(frame);
}

TEST(PciSphSolver2, Parameters) {
    PciSphSolver2 solver;

    solver.setMaxDensityErrorRatio(5.0);
    EXPECT_DOUBLE_EQ(5.0, solver.maxDensityErrorRatio());

    solver.setMaxDensityErrorRatio(-1.0);
    EXPECT_DOUBLE_EQ(0.0, solver.maxDensityErrorRatio());

    solver.setMaxNumberOfIterations(10);
    EXPECT_DOUBLE_EQ(10, solver.maxNumberOfIterations());
}
