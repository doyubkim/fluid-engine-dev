// Copyright (c) 2017 Doyub Kim

#include <unit_tests_utils.h>
#include <jet/box2.h>
#include <jet/custom_implicit_surface2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(CustomImplicitSurface2, SignedDistance) {
    CustomImplicitSurface2 cis(
        [](const Vector2D& pt) {
            return (pt - Vector2D(0.5, 0.5)).length() - 0.25;
        },
        BoundingBox2D({0, 0}, {1, 1}),
        1e-3);

    EXPECT_DOUBLE_EQ(0.25, cis.signedDistance({1, 0.5}));
    EXPECT_DOUBLE_EQ(-0.25, cis.signedDistance({0.5, 0.5}));
    EXPECT_DOUBLE_EQ(0.0, cis.signedDistance({0.5, 0.75}));
}

TEST(CustomImplicitSurface2, CloseestPoint) {
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 0.45})
        .withRadius(0.3)
        .makeShared();
    SurfaceToImplicit2 refSurf(sphere);

    CustomImplicitSurface2 cis1(
        [&](const Vector2D& pt) {
            return refSurf.signedDistance(pt);
        },
        BoundingBox2D({0, 0}, {1, 1}),
        1e-3);

    for (auto sample : kSamplePoints2) {
        if ((sample - sphere->center).length() > 0.01) {
            auto refAns = refSurf.closestPoint(sample);
            auto actAns = cis1.closestPoint(sample);

            EXPECT_VECTOR2_NEAR(refAns, actAns, 1e-3);
        }
    }
}

TEST(CustomImplicitSurface2, CloseestNormal) {
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 0.45})
        .withRadius(0.3)
        .makeShared();
    SurfaceToImplicit2 refSurf(sphere);

    CustomImplicitSurface2 cis1(
        [&](const Vector2D& pt) {
            return refSurf.signedDistance(pt);
        },
        BoundingBox2D({0, 0}, {1, 1}),
        1e-3);

    for (auto sample : kSamplePoints2) {
        auto refAns = refSurf.closestNormal(sample);
        auto actAns = cis1.closestNormal(sample);

        EXPECT_VECTOR2_NEAR(refAns, actAns, 1e-3);
    }
}

TEST(CustomImplicitSurface2, Intersects) {
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 0.45})
        .withRadius(0.3)
        .makeShared();
    SurfaceToImplicit2 refSurf(sphere);

    CustomImplicitSurface2 cis1(
        [&](const Vector2D& pt) {
            return refSurf.signedDistance(pt);
        },
        BoundingBox2D({0, 0}, {1, 1}),
        1e-3);

    size_t n = sizeof(kSamplePoints2) / sizeof(kSamplePoints2[0]);
    for (size_t i = 0; i < n; ++i) {
        auto x = kSamplePoints2[i];
        auto d = kSampleDirs2[i];
        bool refAns = refSurf.intersects(Ray2D(x, d));
        bool actAns = cis1.intersects(Ray2D(x, d));
        EXPECT_EQ(refAns, actAns);
    }
}

TEST(CustomImplicitSurface2, ClosestIntersection) {
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 0.45})
        .withRadius(0.3)
        .makeShared();
    SurfaceToImplicit2 refSurf(sphere);

    CustomImplicitSurface2 cis1(
        [&](const Vector2D& pt) {
            return refSurf.signedDistance(pt);
        },
        BoundingBox2D({0, 0}, {1, 1}),
        1e-3);

    size_t n = sizeof(kSamplePoints2) / sizeof(kSamplePoints2[0]);
    for (size_t i = 0; i < n; ++i) {
        auto x = kSamplePoints2[i];
        auto d = kSampleDirs2[i];
        auto refAns = refSurf.closestIntersection(Ray2D(x, d));
        auto actAns = cis1.closestIntersection(Ray2D(x, d));
        EXPECT_EQ(refAns.isIntersecting, actAns.isIntersecting);
        EXPECT_NEAR(refAns.t, actAns.t, 1e-2);
        EXPECT_VECTOR2_NEAR(refAns.point, actAns.point, 1e-2);
        EXPECT_VECTOR2_NEAR(refAns.normal, actAns.normal, 1e-2);
    }
}
