// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <perf_tests.h>

#include <jet/timer.h>
#include <jet/triangle_mesh3.h>

#include <gtest/gtest.h>

#include <random>

using namespace jet;

class TriangleMesh3Tests : public ::testing::Test {
protected:
    TriangleMesh3 triMesh;

    virtual void SetUp() {
        std::ifstream file(RESOURCES_DIR "bunny.obj");
        ASSERT_TRUE(file);

        if (file) {
            triMesh.readObj(&file);
            file.close();
        }
    }
};

TEST_F(TriangleMesh3Tests, ClosestPoint) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<> d(0.0, 1.0);
    const auto makeVec = [&]() { return Vector3D(d(rng), d(rng), d(rng)); };

    std::vector<Vector3D> results;

    Timer timer;
    size_t n = 1000;
    for (size_t i = 0; i < n; ++i) {
        results.push_back(triMesh.closestPoint(makeVec()));
    }

    JET_PRINT_INFO("TriangleMesh3::closestPoint calls took %f sec.\n",
                   timer.durationInSeconds());

    EXPECT_EQ(n, results.size());
}