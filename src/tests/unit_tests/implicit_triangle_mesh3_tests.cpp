// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

    std::ifstream objFile(RESOURCES_DIR "/cube.obj");
    auto mesh = TriangleMesh3::builder().makeShared();
    mesh->readObj(&objFile);

    auto imesh = ImplicitTriangleMesh3::builder()
        .withTriangleMesh(mesh)
        .withResolutionX(20)
        .makeShared();

    for (size_t i = 0; i < getNumberOfSamplePoints3(); ++i) {
        auto sample = getSamplePoints3()[i];
        auto refAns = refSurf.signedDistance(sample);
        auto actAns = imesh->signedDistance(sample);

        EXPECT_NEAR(refAns, actAns, 1.0 / 20);
    }
}
