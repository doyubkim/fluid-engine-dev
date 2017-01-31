// Copyright (c) 2017 Doyub Kim

#include <unit_tests_utils.h>
#include <jet/box3.h>
#include <jet/implicit_triangle_mesh3.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(ImplicitTriangleMesh3, SignedDistance) {
    auto box = Box3::builder()
        .withLowerCorner({0, 0, 0})
        .withUpperCorner({1, 1, 1})
        .makeShared();
    SurfaceToImplicit3 refSurf(box);

    std::ifstream objFile("resources/cube.obj");
    auto mesh = TriangleMesh3::builder().makeShared();
    mesh->readObj(&objFile);

    auto imesh = ImplicitTriangleMesh3::builder()
        .withTriangleMesh(mesh)
        .withResolutionX(20)
        .makeShared();

    for (auto sample : kSamplePoints3) {
        auto refAns = refSurf.signedDistance(sample);
        auto actAns = imesh->signedDistance(sample);

        EXPECT_NEAR(refAns, actAns, 1.0 / 20);
    }
}
