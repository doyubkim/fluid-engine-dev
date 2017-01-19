// Copyright (c) 2017 Doyub Kim

#include <jet/pbf_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PbfSolver2, UpdateEmpty) {
    // Empty solver test
    PbfSolver2 solver;
    Frame frame(1, 0.01);
    solver.update(frame);
    solver.update(frame);
}

TEST(PbfSolver2, Parameters) {
    PbfSolver2 solver;

    solver.setPseudoViscosityCoefficient(0.3);
    EXPECT_DOUBLE_EQ(0.3, solver.pseudoViscosityCoefficient());

    solver.setPseudoViscosityCoefficient(-1.0);
    EXPECT_DOUBLE_EQ(0.0, solver.pseudoViscosityCoefficient());

    solver.setPseudoViscosityCoefficient(5.0);
    EXPECT_DOUBLE_EQ(1.0, solver.pseudoViscosityCoefficient());

    solver.setMaxNumberOfIterations(7);
    EXPECT_EQ(7u, solver.maxNumberOfIterations());

    solver.setLambdaRelaxation(0.1);
    EXPECT_DOUBLE_EQ(0.1, solver.lambdaRelaxation());

    solver.setAntiClusteringDenominatorFactor(0.3);
    EXPECT_DOUBLE_EQ(0.3, solver.antiClusteringDenominatorFactor());

    solver.setAntiClusteringStrength(1e-3);
    EXPECT_DOUBLE_EQ(1e-3, solver.antiClusteringStrength());

    solver.setAntiClusteringExponent(3.0);
    EXPECT_DOUBLE_EQ(3.0, solver.antiClusteringExponent());

    EXPECT_TRUE(solver.sphSystemData() != nullptr);
}
