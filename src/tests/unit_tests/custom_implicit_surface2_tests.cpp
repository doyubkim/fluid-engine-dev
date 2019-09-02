// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/box2.h>
#include <jet/custom_implicit_surface2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

TEST(CustomImplicitSurface2, SignedDistance) {
    CustomImplicitSurface2 cis(
        [](const Vector2D& pt) {
            return (pt - Vector2D(0.5, 0.5)).length() - 0.25;
        },
        BoundingBox2D({0, 0}, {1, 1}), 1e-3);

    EXPECT_DOUBLE_EQ(0.25, cis.signedDistance({1, 0.5}));
    EXPECT_DOUBLE_EQ(-0.25, cis.signedDistance({0.5, 0.5}));
    EXPECT_DOUBLE_EQ(0.0, cis.signedDistance({0.5, 0.75}));
}

TEST(CustomImplicitSurface2, CloseestPoint) {
    auto sphere =
        Sphere2::builder().withCenter({0.5, 0.45}).withRadius(0.3).makeShared();
    SurfaceToImplicit2 refSurf(sphere);

    CustomImplicitSurface2 cis1(
        [&](const Vector2D& pt) { return refSurf.signedDistance(pt); },
        BoundingBox2D({0, 0}, {1, 1}), 1e-3);

    for (size_t i = 0; i < getNumberOfSamplePoints2(); ++i) {
        auto sample = getSamplePoints2()[i];
        if ((sample - sphere->center).length() > 0.01) {
            auto refAns = refSurf.closestPoint(sample);
            auto actAns = cis1.closestPoint(sample);

            EXPECT_VECTOR2_NEAR(refAns, actAns, 1e-3);
        }
    }
}

TEST(CustomImplicitSurface2, CloseestNormal) {
    auto sphere =
        Sphere2::builder().withCenter({0.5, 0.45}).withRadius(0.3).makeShared();
    SurfaceToImplicit2 refSurf(sphere);

    CustomImplicitSurface2 cis1(
        [&](const Vector2D& pt) { return refSurf.signedDistance(pt); },
        BoundingBox2D({0, 0}, {1, 1}), 1e-3);

    for (size_t i = 0; i < getNumberOfSamplePoints2(); ++i) {
        auto sample = getSamplePoints2()[i];
        auto refAns = refSurf.closestNormal(sample);
        auto actAns = cis1.closestNormal(sample);

        EXPECT_VECTOR2_NEAR(refAns, actAns, 1e-3);
    }
}

TEST(CustomImplicitSurface2, Intersects) {
    auto sphere =
        Sphere2::builder().withCenter({0.5, 0.45}).withRadius(0.3).makeShared();
    SurfaceToImplicit2 refSurf(sphere);

    CustomImplicitSurface2 cis1(
        [&](const Vector2D& pt) { return refSurf.signedDistance(pt); },
        BoundingBox2D({0, 0}, {1, 1}), 1e-3);

    for (size_t i = 0; i < getNumberOfSamplePoints2(); ++i) {
        auto x = getSamplePoints2()[i];
        auto d = getSampleDirs2()[i];
        bool refAns = refSurf.intersects(Ray2D(x, d));
        bool actAns = cis1.intersects(Ray2D(x, d));
        EXPECT_EQ(refAns, actAns);
    }
}

TEST(CustomImplicitSurface2, ClosestIntersection) {
    auto sphere =
        Sphere2::builder().withCenter({0.5, 0.45}).withRadius(0.3).makeShared();
    SurfaceToImplicit2 refSurf(sphere);

    CustomImplicitSurface2 cis1(
        [&](const Vector2D& pt) { return refSurf.signedDistance(pt); },
        BoundingBox2D({0, 0}, {1, 1}), 1e-3, 1e-3);

    for (size_t i = 0; i < getNumberOfSamplePoints2(); ++i) {
        auto x = getSamplePoints2()[i];
        auto d = getSampleDirs2()[i];
        auto refAns = refSurf.closestIntersection(Ray2D(x, d));
        auto actAns = cis1.closestIntersection(Ray2D(x, d));
        EXPECT_EQ(refAns.isIntersecting, actAns.isIntersecting);
        EXPECT_NEAR(refAns.distance, actAns.distance, 1e-5);
        EXPECT_VECTOR2_NEAR(refAns.point, actAns.point, 1e-5);
        EXPECT_VECTOR2_NEAR(refAns.normal, actAns.normal, 1e-5);
    }
}
