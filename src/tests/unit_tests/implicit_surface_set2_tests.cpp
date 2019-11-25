// Copyright (c) Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "unit_tests_utils.h"

#include <jet/box2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/plane2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(ImplicitSurfaceSet2, Constructor) {
    ImplicitSurfaceSet2 sset;
    EXPECT_EQ(0u, sset.numberOfSurfaces());

    sset.isNormalFlipped = true;
    auto box = std::make_shared<Box2>(BoundingBox2D({0, 0}, {1, 2}));
    sset.addExplicitSurface(box);

    ImplicitSurfaceSet2 sset2(sset);
    EXPECT_EQ(1u, sset2.numberOfSurfaces());
    EXPECT_TRUE(sset2.isNormalFlipped);

    ImplicitSurfaceSet2 sset3({box});
    EXPECT_EQ(1u, sset3.numberOfSurfaces());
}

TEST(ImplicitSurfaceSet2, NumberOfSurfaces) {
    ImplicitSurfaceSet2 sset;

    auto box = std::make_shared<Box2>(BoundingBox2D({0, 0}, {1, 2}));
    sset.addExplicitSurface(box);

    EXPECT_EQ(1u, sset.numberOfSurfaces());
}

TEST(ImplicitSurfaceSet2, SurfaceAt) {
    ImplicitSurfaceSet2 sset;

    auto box1 = std::make_shared<Box2>(BoundingBox2D({0, 0}, {1, 2}));
    auto box2 = std::make_shared<Box2>(BoundingBox2D({3, 4}, {5, 6}));
    sset.addExplicitSurface(box1);
    sset.addExplicitSurface(box2);

    auto implicitSurfaceAt0 =
        std::dynamic_pointer_cast<SurfaceToImplicit2>(sset.surfaceAt(0));
    auto implicitSurfaceAt1 =
        std::dynamic_pointer_cast<SurfaceToImplicit2>(sset.surfaceAt(1));

    EXPECT_EQ(std::dynamic_pointer_cast<Surface2>(box1),
              implicitSurfaceAt0->surface());
    EXPECT_EQ(std::dynamic_pointer_cast<Surface2>(box2),
              implicitSurfaceAt1->surface());
}

TEST(ImplicitSurfaceSet2, AddSurface) {
    ImplicitSurfaceSet2 sset;

    auto box1 = std::make_shared<Box2>(BoundingBox2D({0, 0}, {1, 2}));
    auto box2 = std::make_shared<Box2>(BoundingBox2D({3, 4}, {5, 6}));
    auto implicitBox = std::make_shared<SurfaceToImplicit2>(box2);

    sset.addExplicitSurface(box1);
    sset.addSurface(implicitBox);

    EXPECT_EQ(2u, sset.numberOfSurfaces());

    auto implicitSurfaceAt0 =
        std::dynamic_pointer_cast<SurfaceToImplicit2>(sset.surfaceAt(0));
    auto implicitSurfaceAt1 =
        std::dynamic_pointer_cast<SurfaceToImplicit2>(sset.surfaceAt(1));

    EXPECT_EQ(std::dynamic_pointer_cast<Surface2>(box1),
              implicitSurfaceAt0->surface());
    EXPECT_EQ(implicitBox, implicitSurfaceAt1);
}

TEST(ImplicitSurfaceSet2, ClosestPoint) {
    BoundingBox2D bbox(Vector2D(), Vector2D(1, 2));

    Box2Ptr box = std::make_shared<Box2>(bbox);
    box->isNormalFlipped = true;

    ImplicitSurfaceSet2Ptr sset = std::make_shared<ImplicitSurfaceSet2>();
    Vector2D emptyPoint = sset->closestPoint({1.0, 2.0});
    EXPECT_DOUBLE_EQ(kMaxD, emptyPoint.x);
    EXPECT_DOUBLE_EQ(kMaxD, emptyPoint.y);

    sset->addExplicitSurface(box);

    Vector2D pt(0.5, 2.5);
    Vector2D boxPoint = box->closestPoint(pt);
    Vector2D setPoint = sset->closestPoint(pt);
    EXPECT_DOUBLE_EQ(boxPoint.x, setPoint.x);
    EXPECT_DOUBLE_EQ(boxPoint.y, setPoint.y);
}

TEST(ImplicitSurfaceSet2, ClosestDistance) {
    BoundingBox2D bbox(Vector2D(), Vector2D(1, 2));

    Box2Ptr box = std::make_shared<Box2>(bbox);
    box->isNormalFlipped = true;

    ImplicitSurfaceSet2Ptr sset = std::make_shared<ImplicitSurfaceSet2>();
    sset->addExplicitSurface(box);

    Vector2D pt(0.5, 2.5);
    double boxDist = box->closestDistance(pt);
    double setDist = sset->closestDistance(pt);
    EXPECT_DOUBLE_EQ(boxDist, setDist);
}

TEST(ImplicitSurfaceSet2, Intersects) {
    ImplicitSurfaceSet2 sset;
    auto box = std::make_shared<Box2>(BoundingBox2D({-1, 2}, {5, 3}));
    sset.addExplicitSurface(box);

    bool result0 =
        sset.intersects(Ray2D(Vector2D(1, 4), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result0);

    bool result1 =
        sset.intersects(Ray2D(Vector2D(1, 2.5), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result1);

    bool result2 =
        sset.intersects(Ray2D(Vector2D(1, 1), Vector2D(-1, -1).normalized()));
    EXPECT_FALSE(result2);
}

TEST(ImplicitSurfaceSet2, ClosestIntersection) {
    ImplicitSurfaceSet2 sset;
    auto box = std::make_shared<Box2>(BoundingBox2D({-1, 2}, {5, 3}));
    sset.addExplicitSurface(box);

    SurfaceRayIntersection2 result0 = sset.closestIntersection(
        Ray2D(Vector2D(1, 4), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result0.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(2), result0.distance);
    EXPECT_EQ(Vector2D(0, 3), result0.point);

    SurfaceRayIntersection2 result1 = sset.closestIntersection(
        Ray2D(Vector2D(1, 2.5), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result1.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(0.5), result1.distance);
    EXPECT_EQ(Vector2D(0.5, 2), result1.point);

    SurfaceRayIntersection2 result2 = sset.closestIntersection(
        Ray2D(Vector2D(1, 1), Vector2D(-1, -1).normalized()));
    EXPECT_FALSE(result2.isIntersecting);
}

TEST(ImplicitSurfaceSet2, BoundingBox) {
    ImplicitSurfaceSet2 sset;

    auto box1 = std::make_shared<Box2>(BoundingBox2D({0, 0}, {1, 2}));
    auto box2 = std::make_shared<Box2>(BoundingBox2D({3, 4}, {5, 6}));
    sset.addExplicitSurface(box1);
    sset.addExplicitSurface(box2);

    auto bbox = sset.boundingBox();
    EXPECT_DOUBLE_EQ(0.0, bbox.lowerCorner.x);
    EXPECT_DOUBLE_EQ(0.0, bbox.lowerCorner.y);
    EXPECT_DOUBLE_EQ(5.0, bbox.upperCorner.x);
    EXPECT_DOUBLE_EQ(6.0, bbox.upperCorner.y);
}

TEST(ImplicitSurfaceSet2, SignedDistance) {
    BoundingBox2D bbox(Vector2D(1, 4), Vector2D(5, 6));

    Box2Ptr box = std::make_shared<Box2>(bbox);
    auto implicitBox = std::make_shared<SurfaceToImplicit2>(box);

    ImplicitSurfaceSet2Ptr sset = std::make_shared<ImplicitSurfaceSet2>();
    sset->addExplicitSurface(box);

    Vector2D pt(-1, 7);
    double boxDist = implicitBox->signedDistance(pt);
    double setDist = sset->signedDistance(pt);
    EXPECT_DOUBLE_EQ(boxDist, setDist);
}

TEST(ImplicitSurfaceSet2, ClosestNormal) {
    BoundingBox2D bbox(Vector2D(), Vector2D(1, 2));

    Box2Ptr box = std::make_shared<Box2>(bbox);
    box->isNormalFlipped = true;

    ImplicitSurfaceSet2Ptr sset = std::make_shared<ImplicitSurfaceSet2>();
    Vector2D emptyNormal = sset->closestNormal({1.0, 2.0});
    // No expected value -- just see if it doesn't crash
    (void)emptyNormal;

    sset->addExplicitSurface(box);

    Vector2D pt(0.5, 2.5);
    Vector2D boxNormal = box->closestNormal(pt);
    Vector2D setNormal = sset->closestNormal(pt);
    EXPECT_DOUBLE_EQ(boxNormal.x, setNormal.x);
    EXPECT_DOUBLE_EQ(boxNormal.y, setNormal.y);
}

TEST(ImplicitSurfaceSet2, MixedBoundTypes) {
    BoundingBox2D domain(Vector2D(), Vector2D(1, 2));

    auto plane = Plane2::builder()
                     .withNormal({0, 1})
                     .withPoint({0.0, 0.25 * domain.height()})
                     .makeShared();

    auto sphere = Sphere2::builder()
                      .withCenter(domain.midPoint())
                      .withRadius(0.15 * domain.width())
                      .makeShared();

    auto surfaceSet = ImplicitSurfaceSet2::builder()
                          .withExplicitSurfaces({plane, sphere})
                          .makeShared();

    EXPECT_FALSE(surfaceSet->isBounded());

    auto cp = surfaceSet->closestPoint(Vector2D(0.5, 0.4));
    Vector2D answer(0.5, 0.5);

    EXPECT_VECTOR2_NEAR(answer, cp, 1e-9);
}

TEST(ImplicitSurfaceSet2, IsValidGeometry) {
    auto surfaceSet = ImplicitSurfaceSet2::builder().makeShared();

    EXPECT_FALSE(surfaceSet->isValidGeometry());

    auto box = std::make_shared<Box2>(BoundingBox2D({0, 0}, {1, 2}));
    auto surfaceSet2 = ImplicitSurfaceSet2::builder().makeShared();
    surfaceSet2->addExplicitSurface(box);

    EXPECT_TRUE(surfaceSet2->isValidGeometry());

    surfaceSet2->addSurface(surfaceSet);

    EXPECT_FALSE(surfaceSet2->isValidGeometry());
}

TEST(ImplicitSurfaceSet2, IsInside) {
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

    auto surfaceSet = ImplicitSurfaceSet2::builder()
                          .withExplicitSurfaces({plane, sphere})
                          .withTransform(Transform2(offset, 0.0))
                          .makeShared();

    EXPECT_TRUE(surfaceSet->isInside(Vector2D(0.5, 0.25) + offset));
    EXPECT_TRUE(surfaceSet->isInside(Vector2D(0.5, 1.0) + offset));
    EXPECT_FALSE(surfaceSet->isInside(Vector2D(0.5, 1.5) + offset));
}

TEST(ImplicitSurfaceSet2, UpdateQueryEngine) {
    auto sphere =
        Sphere2::builder().withCenter({-1.0, 1.0}).withRadius(0.5).makeShared();

    auto surfaceSet = ImplicitSurfaceSet2::builder()
                          .withExplicitSurfaces({sphere})
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
    surfaceSet->addExplicitSurface(plane);
    surfaceSet->updateQueryEngine();
    auto bbox4 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox2D({-4.5, 5.5}, {-3.5, 6.5}), bbox4);
}
