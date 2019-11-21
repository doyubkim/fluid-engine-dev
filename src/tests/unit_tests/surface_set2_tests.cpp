// Copyright (c) Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/plane2.h>
#include <jet/sphere2.h>
#include <jet/surface_set2.h>

using namespace jet;

TEST(SurfaceSet2, Constructors) {
    SurfaceSet2 sset1;
    EXPECT_EQ(0u, sset1.numberOfSurfaces());

    auto sph1 =
        Sphere2::builder().withRadius(1.0).withCenter({0, 0}).makeShared();
    auto sph2 =
        Sphere2::builder().withRadius(0.5).withCenter({0, 3}).makeShared();
    auto sph3 =
        Sphere2::builder().withRadius(0.25).withCenter({-2, 0}).makeShared();
    SurfaceSet2 sset2({sph1, sph2, sph3}, Transform2(), false);
    EXPECT_EQ(3u, sset2.numberOfSurfaces());
    EXPECT_EQ(sph1->radius,
              std::dynamic_pointer_cast<Sphere2>(sset2.surfaceAt(0))->radius);
    EXPECT_EQ(sph2->radius,
              std::dynamic_pointer_cast<Sphere2>(sset2.surfaceAt(1))->radius);
    EXPECT_EQ(sph3->radius,
              std::dynamic_pointer_cast<Sphere2>(sset2.surfaceAt(2))->radius);
    EXPECT_EQ(Vector2D(), sset2.transform.translation());
    EXPECT_EQ(0.0, sset2.transform.orientation());

    SurfaceSet2 sset3({sph1, sph2, sph3}, Transform2(Vector2D(1, 2), 0.5),
                      false);
    EXPECT_EQ(Vector2D(1, 2), sset3.transform.translation());
    EXPECT_EQ(0.5, sset3.transform.orientation());

    SurfaceSet2 sset4(sset3);
    EXPECT_EQ(3u, sset4.numberOfSurfaces());
    EXPECT_EQ(sph1->radius,
              std::dynamic_pointer_cast<Sphere2>(sset4.surfaceAt(0))->radius);
    EXPECT_EQ(sph2->radius,
              std::dynamic_pointer_cast<Sphere2>(sset4.surfaceAt(1))->radius);
    EXPECT_EQ(sph3->radius,
              std::dynamic_pointer_cast<Sphere2>(sset4.surfaceAt(2))->radius);
    EXPECT_EQ(Vector2D(1, 2), sset4.transform.translation());
    EXPECT_EQ(0.5, sset4.transform.orientation());
}

TEST(SurfaceSet2, AddSurface) {
    SurfaceSet2 sset1;
    EXPECT_EQ(0u, sset1.numberOfSurfaces());

    auto sph1 =
        Sphere2::builder().withRadius(1.0).withCenter({0, 0}).makeShared();
    auto sph2 =
        Sphere2::builder().withRadius(0.5).withCenter({0, 3}).makeShared();
    auto sph3 =
        Sphere2::builder().withRadius(0.25).withCenter({-2, 0}).makeShared();

    sset1.addSurface(sph1);
    sset1.addSurface(sph2);
    sset1.addSurface(sph3);

    EXPECT_EQ(3u, sset1.numberOfSurfaces());
    EXPECT_EQ(sph1->radius,
              std::dynamic_pointer_cast<Sphere2>(sset1.surfaceAt(0))->radius);
    EXPECT_EQ(sph2->radius,
              std::dynamic_pointer_cast<Sphere2>(sset1.surfaceAt(1))->radius);
    EXPECT_EQ(sph3->radius,
              std::dynamic_pointer_cast<Sphere2>(sset1.surfaceAt(2))->radius);
    EXPECT_EQ(Vector2D(), sset1.transform.translation());
    EXPECT_EQ(0.0, sset1.transform.orientation());
}

TEST(SurfaceSet2, ClosestPoint) {
    SurfaceSet2 sset1;

    // Test empty set
    Vector2D emptyPoint = sset1.closestPoint({1.0, 2.0});
    EXPECT_DOUBLE_EQ(kMaxD, emptyPoint.x);
    EXPECT_DOUBLE_EQ(kMaxD, emptyPoint.y);

    size_t numSamples = getNumberOfSamplePoints2();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints2()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceSearch = [&](const Vector2D& pt) {
        double minDist2 = kMaxD;
        Vector2D result;
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
        auto actual = sset1.closestPoint(getSamplePoints2()[i]);
        auto expected = bruteForceSearch(getSamplePoints2()[i]);
        EXPECT_VECTOR2_EQ(expected, actual);
    }

    // Now with translation instead of center
    SurfaceSet2 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints2()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset2.closestPoint(getSamplePoints2()[i]);
        auto expected = bruteForceSearch(getSamplePoints2()[i]);
        EXPECT_VECTOR2_EQ(expected, actual);
    }
}

TEST(SurfaceSet2, ClosestNormal) {
    SurfaceSet2 sset1;

    // Test empty set
    Vector2D emptyNormal = sset1.closestNormal({1.0, 2.0});
    // No expected value -- just see if it doesn't crash
    (void)emptyNormal;

    size_t numSamples = getNumberOfSamplePoints2();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints2()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceSearch = [&](const Vector2D& pt) {
        double minDist2 = kMaxD;
        Vector2D result;
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
        auto actual = sset1.closestNormal(getSamplePoints2()[i]);
        auto expected = bruteForceSearch(getSamplePoints2()[i]);
        EXPECT_VECTOR2_EQ(expected, actual);
    }

    // Now with translation instead of center
    SurfaceSet2 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints2()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset2.closestNormal(getSamplePoints2()[i]);
        auto expected = bruteForceSearch(getSamplePoints2()[i]);
        EXPECT_VECTOR2_EQ(expected, actual);
    }
}

TEST(SurfaceSet2, ClosestDistance) {
    SurfaceSet2 sset1;

    size_t numSamples = getNumberOfSamplePoints2();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints2()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceSearch = [&](const Vector2D& pt) {
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
        auto actual = sset1.closestDistance(getSamplePoints2()[i]);
        auto expected = bruteForceSearch(getSamplePoints2()[i]);
        EXPECT_DOUBLE_EQ(expected, actual);
    }

    // Now with translation instead of center
    SurfaceSet2 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints2()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        auto actual = sset2.closestDistance(getSamplePoints2()[i]);
        auto expected = bruteForceSearch(getSamplePoints2()[i]);
        EXPECT_DOUBLE_EQ(expected, actual);
    }
}

TEST(SurfaceSet2, Intersects) {
    SurfaceSet2 sset1;

    size_t numSamples = getNumberOfSamplePoints2();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints2()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceTest = [&](const Ray2D& ray) {
        for (size_t i = 0; i < numSamples / 2; ++i) {
            if (sset1.surfaceAt(i)->intersects(ray)) {
                return true;
            }
        }
        return false;
    };

    // Use second half of the samples as the query points
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        Ray2D ray(getSamplePoints2()[i], getSampleDirs2()[i]);
        bool actual = sset1.intersects(ray);
        bool expected = bruteForceTest(ray);
        EXPECT_EQ(expected, actual);
    }

    // Now with translation instead of center
    SurfaceSet2 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints2()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        Ray2D ray(getSamplePoints2()[i], getSampleDirs2()[i]);
        bool actual = sset2.intersects(ray);
        bool expected = bruteForceTest(ray);
        EXPECT_EQ(expected, actual);
    }
}

TEST(SurfaceSet2, ClosestIntersection) {
    SurfaceSet2 sset1;

    size_t numSamples = getNumberOfSamplePoints2();

    // Use first half of the samples as the centers of the spheres
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints2()[i])
                       .makeShared();
        sset1.addSurface(sph);
    }

    const auto bruteForceTest = [&](const Ray2D& ray) {
        SurfaceRayIntersection2 result{};
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
        Ray2D ray(getSamplePoints2()[i], getSampleDirs2()[i]);
        auto actual = sset1.closestIntersection(ray);
        auto expected = bruteForceTest(ray);
        EXPECT_DOUBLE_EQ(expected.distance, actual.distance);
        EXPECT_VECTOR2_EQ(expected.point, actual.point);
        EXPECT_VECTOR2_EQ(expected.normal, actual.normal);
        EXPECT_EQ(expected.isIntersecting, actual.isIntersecting);
    }

    // Now with translation instead of center
    SurfaceSet2 sset2;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints2()[i])
                       .makeShared();
        sset2.addSurface(sph);
    }
    for (size_t i = numSamples / 2; i < numSamples; ++i) {
        Ray2D ray(getSamplePoints2()[i], getSampleDirs2()[i]);
        auto actual = sset2.closestIntersection(ray);
        auto expected = bruteForceTest(ray);
        EXPECT_DOUBLE_EQ(expected.distance, actual.distance);
        EXPECT_VECTOR2_EQ(expected.point, actual.point);
        EXPECT_VECTOR2_EQ(expected.normal, actual.normal);
        EXPECT_EQ(expected.isIntersecting, actual.isIntersecting);
    }
}

TEST(SurfaceSet2, BoundingBox) {
    SurfaceSet2 sset1;

    size_t numSamples = getNumberOfSamplePoints2();

    // Use first half of the samples as the centers of the spheres
    BoundingBox2D answer;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter(getSamplePoints2()[i])
                       .makeShared();
        sset1.addSurface(sph);

        answer.merge(sph->boundingBox());
    }

    EXPECT_BOUNDING_BOX2_NEAR(answer, sset1.boundingBox(), 1e-9);

    // Now with translation instead of center
    SurfaceSet2 sset2;
    BoundingBox2D debug;
    for (size_t i = 0; i < numSamples / 2; ++i) {
        auto sph = Sphere2::builder()
                       .withRadius(0.01)
                       .withCenter({0, 0, 0})
                       .withTranslation(getSamplePoints2()[i])
                       .makeShared();
        sset2.addSurface(sph);

        debug.merge(sph->boundingBox());
    }

    EXPECT_BOUNDING_BOX2_NEAR(answer, debug, 1e-9);
    EXPECT_BOUNDING_BOX2_NEAR(answer, sset2.boundingBox(), 1e-9);
}

TEST(SurfaceSet2, MixedBoundTypes) {
    BoundingBox2D domain(Vector2D(), Vector2D(1, 2));

    auto plane = Plane2::builder()
                     .withNormal({0, 1})
                     .withPoint({0.0, 0.25 * domain.height()})
                     .makeShared();

    auto sphere = Sphere2::builder()
                      .withCenter(domain.midPoint())
                      .withRadius(0.15 * domain.width())
                      .makeShared();

    auto surfaceSet =
        SurfaceSet2::builder().withSurfaces({plane, sphere}).makeShared();

    EXPECT_FALSE(surfaceSet->isBounded());

    auto cp = surfaceSet->closestPoint(Vector2D(0.5, 0.4));
    Vector2D answer(0.5, 0.5);

    EXPECT_VECTOR2_NEAR(answer, cp, 1e-9);
}

TEST(SurfaceSet2, IsValidGeometry) {
    auto surfaceSet = SurfaceSet2::builder().makeShared();

    EXPECT_FALSE(surfaceSet->isValidGeometry());

    BoundingBox2D domain(Vector2D(), Vector2D(1, 2));

    auto plane = Plane2::builder()
                     .withNormal({0, 1})
                     .withPoint({0, 0.25 * domain.height()})
                     .makeShared();

    auto sphere = Sphere2::builder()
                      .withCenter(domain.midPoint())
                      .withRadius(0.15 * domain.width())
                      .makeShared();

    auto surfaceSet2 =
        SurfaceSet2::builder().withSurfaces({plane, sphere}).makeShared();

    EXPECT_TRUE(surfaceSet2->isValidGeometry());

    surfaceSet2->addSurface(surfaceSet);

    EXPECT_FALSE(surfaceSet2->isValidGeometry());
}

TEST(SurfaceSet2, IsInside) {
    BoundingBox2D domain(Vector2D(), Vector2D(1, 2));
    Vector2D offset(1, 2);

    auto plane = Plane2::builder()
                     .withNormal({0, 1})
                     .withPoint({0, 0.25 * domain.height()})
                     .makeShared();

    auto sphere = Sphere2::builder()
                      .withCenter(domain.midPoint())
                      .withRadius(0.15 * domain.width())
                      .makeShared();

    auto surfaceSet = SurfaceSet2::builder()
                          .withSurfaces({plane, sphere})
                          .withTransform(Transform2(offset, 0.0))
                          .makeShared();

    EXPECT_TRUE(surfaceSet->isInside(Vector2D(0.5, 0.25) + offset));
    EXPECT_TRUE(surfaceSet->isInside(Vector2D(0.5, 1.0) + offset));
    EXPECT_FALSE(surfaceSet->isInside(Vector2D(0.5, 1.5) + offset));
}

TEST(SurfaceSet2, UpdateQueryEngine) {
    auto sphere =
        Sphere2::builder().withCenter({-1.0, 1.0}).withRadius(0.5).makeShared();

    auto surfaceSet = SurfaceSet2::builder()
                          .withSurfaces({sphere})
                          .withTransform(Transform2({1.0, 2.0}, 0.0))
                          .makeShared();

    auto bbox1 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox2D({-0.5, 2.5}, {0.5, 3.5}), bbox1);

    surfaceSet->transform = Transform2({3.0, -4.0}, 0.0);
    surfaceSet->updateQueryEngine();
    auto bbox2 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox2D({1.5, -3.5}, {2.5, -2.5}), bbox2);

    sphere->transform = Transform2({-6.0, 9.0}, 0.0);
    surfaceSet->updateQueryEngine();
    auto bbox3 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox2D({-4.5, 5.5}, {-3.5, 6.5}), bbox3);

    // Plane is unbounded. Total bbox should ignore it.
    auto plane = Plane2::builder().withNormal({1.0, 0.0}).makeShared();
    surfaceSet->addSurface(plane);
    surfaceSet->updateQueryEngine();
    auto bbox4 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox2D({-4.5, 5.5}, {-3.5, 6.5}), bbox4);
}
