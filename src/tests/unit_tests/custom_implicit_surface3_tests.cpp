// Copyright (c) 2017 Doyub Kim

#include <unit_tests_utils.h>
#include <jet/box3.h>
#include <jet/custom_implicit_surface3.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(CustomImplicitSurface3, SignedDistance) {
    CustomImplicitSurface3 cis(
        [](const Vector3D& pt) {
            return (pt - Vector3D(0.5, 0.5, 0.5)).length() - 0.25;
        },
        BoundingBox3D({0, 0, 0}, {1, 1, 1}),
        1e-3);

    EXPECT_DOUBLE_EQ(0.25, cis.signedDistance({1, 0.5, 0.5}));
    EXPECT_DOUBLE_EQ(-0.25, cis.signedDistance({0.5, 0.5, 0.5}));
    EXPECT_DOUBLE_EQ(0.0, cis.signedDistance({0.5, 0.75, 0.5}));
}

TEST(CustomImplicitSurface3, CloseestPoint) {
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.45, 0.55})
        .withRadius(0.3)
        .makeShared();
    SurfaceToImplicit3 refSurf(sphere);

    CustomImplicitSurface3 cis1(
        [&](const Vector3D& pt) {
            return refSurf.signedDistance(pt);
        },
        BoundingBox3D({0, 0, 0}, {1, 1, 1}),
        1e-3);

    for (auto sample : kSamplePoints3) {
        if ((sample - sphere->center).length() > 0.01) {
            auto refAns = refSurf.closestPoint(sample);
            auto actAns = cis1.closestPoint(sample);

            EXPECT_VECTOR3_NEAR(refAns, actAns, 1e-3);
        }
    }
}

TEST(CustomImplicitSurface3, CloseestNormal) {
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.45, 0.55})
        .withRadius(0.3)
        .makeShared();
    SurfaceToImplicit3 refSurf(sphere);

    CustomImplicitSurface3 cis1(
        [&](const Vector3D& pt) {
            return refSurf.signedDistance(pt);
        },
        BoundingBox3D({0, 0, 0}, {1, 1, 1}),
        1e-3);

    for (auto sample : kSamplePoints3) {
        auto refAns = refSurf.closestNormal(sample);
        auto actAns = cis1.closestNormal(sample);

        EXPECT_VECTOR3_NEAR(refAns, actAns, 1e-3);
    }
}

TEST(CustomImplicitSurface3, Intersects) {
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.45, 0.55})
        .withRadius(0.3)
        .makeShared();
    SurfaceToImplicit3 refSurf(sphere);

    CustomImplicitSurface3 cis1(
        [&](const Vector3D& pt) {
            return refSurf.signedDistance(pt);
        },
        BoundingBox3D({0, 0, 0}, {1, 1, 1}),
        1e-3);

    size_t n = sizeof(kSamplePoints3) / sizeof(kSamplePoints3[0]);
    for (size_t i = 0; i < n; ++i) {
        auto x = kSamplePoints3[i];
        auto d = kSampleDirs3[i];
        bool refAns = refSurf.intersects(Ray3D(x, d));
        bool actAns = cis1.intersects(Ray3D(x, d));
        EXPECT_EQ(refAns, actAns);
    }
}

TEST(CustomImplicitSurface3, ClosestIntersection) {
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.45, 0.55})
        .withRadius(0.3)
        .makeShared();
    SurfaceToImplicit3 refSurf(sphere);

    CustomImplicitSurface3 cis1(
        [&](const Vector3D& pt) {
            return refSurf.signedDistance(pt);
        },
        BoundingBox3D({0, 0, 0}, {1, 1, 1}),
        1e-3);

    size_t n = sizeof(kSamplePoints3) / sizeof(kSamplePoints3[0]);
    for (size_t i = 0; i < n; ++i) {
        auto x = kSamplePoints3[i];
        auto d = kSampleDirs3[i];
        auto refAns = refSurf.closestIntersection(Ray3D(x, d));
        auto actAns = cis1.closestIntersection(Ray3D(x, d));
        EXPECT_EQ(refAns.isIntersecting, actAns.isIntersecting);
        EXPECT_NEAR(refAns.t, actAns.t, 1e-2);
        EXPECT_VECTOR3_NEAR(refAns.point, actAns.point, 1e-2);
        EXPECT_VECTOR3_NEAR(refAns.normal, actAns.normal, 1e-2);
    }
}
