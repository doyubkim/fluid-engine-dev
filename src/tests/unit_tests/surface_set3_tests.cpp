// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/sphere3.h>
#include <jet/surface_set3.h>

using namespace jet;

TEST(SurfaceSet3, Constructors) {
    SurfaceSet3 sset1;
    EXPECT_EQ(0u, sset1.numberOfSurfaces());

    auto sph1 =
        Sphere3::builder().withRadius(1.0).withCenter({0, 0, 0}).makeShared();
    auto sph2 =
        Sphere3::builder().withRadius(0.5).withCenter({0, 3, 2}).makeShared();
    auto sph3 =
        Sphere3::builder().withRadius(0.25).withCenter({-2, 0, 0}).makeShared();
    SurfaceSet3 sset2({sph1, sph2, sph3}, Transform3(), false);
    EXPECT_EQ(3u, sset2.numberOfSurfaces());
    EXPECT_EQ(sph1->radius,
              std::dynamic_pointer_cast<Sphere3>(sset2.surfaceAt(0))->radius);
    EXPECT_EQ(sph2->radius,
              std::dynamic_pointer_cast<Sphere3>(sset2.surfaceAt(1))->radius);
    EXPECT_EQ(sph3->radius,
              std::dynamic_pointer_cast<Sphere3>(sset2.surfaceAt(2))->radius);
    EXPECT_EQ(Vector3D(), sset2.transform.translation());
    EXPECT_EQ(QuaternionD(), sset2.transform.orientation());

    SurfaceSet3 sset3(
        {sph1, sph2, sph3},
        Transform3(Vector3D(1, 2, 3), QuaternionD({1, 0, 0}, 0.5)), false);
    EXPECT_EQ(Vector3D(1, 2, 3), sset3.transform.translation());
    EXPECT_EQ(QuaternionD({1, 0, 0}, 0.5), sset3.transform.orientation());
}

TEST(SurfaceSet3, AddSurface) {
    SurfaceSet3 sset1;
    EXPECT_EQ(0u, sset1.numberOfSurfaces());

    auto sph1 =
        Sphere3::builder().withRadius(1.0).withCenter({0, 0, 0}).makeShared();
    auto sph2 =
        Sphere3::builder().withRadius(0.5).withCenter({0, 3, 2}).makeShared();
    auto sph3 =
        Sphere3::builder().withRadius(0.25).withCenter({-2, 0, 0}).makeShared();

    sset1.addSurface(sph1);
    sset1.addSurface(sph2);
    sset1.addSurface(sph3);

    EXPECT_EQ(3u, sset1.numberOfSurfaces());
    EXPECT_EQ(sph1->radius,
              std::dynamic_pointer_cast<Sphere3>(sset1.surfaceAt(0))->radius);
    EXPECT_EQ(sph2->radius,
              std::dynamic_pointer_cast<Sphere3>(sset1.surfaceAt(1))->radius);
    EXPECT_EQ(sph3->radius,
              std::dynamic_pointer_cast<Sphere3>(sset1.surfaceAt(2))->radius);
    EXPECT_EQ(Vector3D(), sset1.transform.translation());
    EXPECT_EQ(QuaternionD(), sset1.transform.orientation());
}

TEST(SurfaceSet3, ClosestPoint) {
    SurfaceSet3 sset1;

    size_t numSamples = getNumberOfSamplePoints3();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints3()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceSearch = [&](const Vector3D& pt) {
        double minDist2 = kMaxD;
        Vector3D result;
        for (size_t i = 0; i < numSamples / 2; ++i) {
            auto localResult = sset1.surfaceAt(i)->closestPoint(pt);
            double localDist2 = pt.distanceSquaredTo(localResult);
            if (localDist2 < minDist2) {
                minDist2 = localDist2;
                result = localResult;
            }
        }
        return result;
    };

    // Use second half of the samples as the query points
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset1.closestPoint(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_VECTOR3_EQ(expected, actual);
    }

    // Now with translation instead of center
    SurfaceSet3 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints3()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset2.closestPoint(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_VECTOR3_EQ(expected, actual);
    }
}

TEST(SurfaceSet3, ClosestNormal) {
    SurfaceSet3 sset1;

    size_t numSamples = getNumberOfSamplePoints3();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints3()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceSearch = [&](const Vector3D& pt) {
        double minDist2 = kMaxD;
        Vector3D result;
        for (size_t i = 0; i < numSamples / 2; ++i) {
            auto localResult = sset1.surfaceAt(i)->closestNormal(pt);
            auto closestPt = sset1.surfaceAt(i)->closestPoint(pt);
            double localDist2 = pt.distanceSquaredTo(closestPt);
            if (localDist2 < minDist2) {
                minDist2 = localDist2;
                result = localResult;
            }
        }
        return result;
    };

    // Use second half of the samples as the query points
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset1.closestNormal(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_VECTOR3_EQ(expected, actual);
    }

    // Now with translation instead of center
    SurfaceSet3 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints3()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset2.closestNormal(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_VECTOR3_EQ(expected, actual);
    }
}

TEST(SurfaceSet3, ClosestDistance) {
    SurfaceSet3 sset1;

    size_t numSamples = getNumberOfSamplePoints3();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints3()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceSearch = [&](const Vector3D& pt) {
        double minDist = kMaxD;
        for (size_t i = 0; i < numSamples / 2; ++i) {
            double localDist = sset1.surfaceAt(i)->closestDistance(pt);
            if (localDist < minDist) {
                minDist = localDist;
            }
        }
        return minDist;
    };

    // Use second half of the samples as the query points
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset1.closestDistance(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_DOUBLE_EQ(expected, actual);
    }

    // Now with translation instead of center
    SurfaceSet3 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints3()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset2.closestDistance(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_DOUBLE_EQ(expected, actual);
    }
}

TEST(SurfaceSet3, Intersects) {
    SurfaceSet3 sset1;

    size_t numSamples = getNumberOfSamplePoints3();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints3()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceTest = [&](const Ray3D& ray) {
        for (size_t i = 0; i < numSamples / 2; ++i) {
            if (sset1.surfaceAt(i)->intersects(ray)) {
                return true;
            }
        }
        return false;
    };

    // Use second half of the samples as the query points
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        Ray3D ray(getSamplePoints3()[i], getSampleDirs3()[i]);
        bool actual = sset1.intersects(ray);
        bool expected = bruteForceTest(ray);
        EXPECT_EQ(expected, actual);
    }

    // Now with translation instead of center
    SurfaceSet3 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints3()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        Ray3D ray(getSamplePoints3()[i], getSampleDirs3()[i]);
        bool actual = sset2.intersects(ray);
        bool expected = bruteForceTest(ray);
        EXPECT_EQ(expected, actual);
    }
}

TEST(SurfaceSet3, ClosestIntersection) {
    SurfaceSet3 sset1;

    size_t numSamples = getNumberOfSamplePoints3();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints3()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceTest = [&](const Ray3D& ray) {
        SurfaceRayIntersection3 result{};
        for (size_t i = 0; i < numSamples / 2; ++i) {
            auto localResult = sset1.surfaceAt(i)->closestIntersection(ray);
            if (localResult.distance < result.distance) {
                result = localResult;
            }
        }
        return result;
    };

    // Use second half of the samples as the query points
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        Ray3D ray(getSamplePoints3()[i], getSampleDirs3()[i]);
        auto actual = sset1.closestIntersection(ray);
        auto expected = bruteForceTest(ray);
        EXPECT_DOUBLE_EQ(expected.distance, actual.distance);
        EXPECT_VECTOR3_EQ(expected.point, actual.point);
        EXPECT_VECTOR3_EQ(expected.normal, actual.normal);
        EXPECT_EQ(expected.isIntersecting, actual.isIntersecting);
    }

    // Now with translation instead of center
    SurfaceSet3 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints3()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        Ray3D ray(getSamplePoints3()[i], getSampleDirs3()[i]);
        auto actual = sset2.closestIntersection(ray);
        auto expected = bruteForceTest(ray);
        EXPECT_DOUBLE_EQ(expected.distance, actual.distance);
        EXPECT_VECTOR3_EQ(expected.point, actual.point);
        EXPECT_VECTOR3_EQ(expected.normal, actual.normal);
        EXPECT_EQ(expected.isIntersecting, actual.isIntersecting);
    }
}

TEST(SurfaceSet3, BoundingBox) {
    SurfaceSet3 sset1;

    size_t numSamples = getNumberOfSamplePoints3();

    // Use first half of the samples as the centers of the spheres
    BoundingBox3D answer;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints3()[i])
                       .makeShared();
        sset1.addSurface(sph);

        answer.merge(sph->boundingBox());
    }

    EXPECT_BOUNDING_BOX3_NEAR(answer, sset1.boundingBox(), 1e-9);

    // Now with translation instead of center
    SurfaceSet3 sset2;
    BoundingBox3D debug;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere3::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints3()[i])
                       .makeShared();
        sset2.addSurface(sph);

        debug.merge(sph->boundingBox());
    }

    EXPECT_BOUNDING_BOX3_NEAR(answer, debug, 1e-9);
    EXPECT_BOUNDING_BOX3_NEAR(answer, sset2.boundingBox(), 1e-9);
}
