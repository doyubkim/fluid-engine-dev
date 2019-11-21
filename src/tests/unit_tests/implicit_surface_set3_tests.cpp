// Copyright (c) Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "unit_tests_utils.h"

#include <jet/box3.h>
#include <jet/implicit_surface_set3.h>
#include <jet/plane3.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(ImplicitSurfaceSet3, Constructor) {
    ImplicitSurfaceSet3 sset;
    EXPECT_EQ(0u, sset.numberOfSurfaces());

    sset.isNormalFlipped = true;
    auto box = std::make_shared<Box3>(BoundingBox3D({0, 0, 0}, {1, 2, 3}));
    sset.addExplicitSurface(box);

    ImplicitSurfaceSet3 sset2(sset);
    EXPECT_EQ(1u, sset2.numberOfSurfaces());
    EXPECT_TRUE(sset2.isNormalFlipped);

    ImplicitSurfaceSet3 sset3({box});
    EXPECT_EQ(1u, sset3.numberOfSurfaces());
}

TEST(ImplicitSurfaceSet3, NumberOfSurfaces) {
    ImplicitSurfaceSet3 sset;

    auto box = std::make_shared<Box3>(BoundingBox3D({0, 0, 0}, {1, 2, 3}));
    sset.addExplicitSurface(box);

    EXPECT_EQ(1u, sset.numberOfSurfaces());
}

TEST(ImplicitSurfaceSet3, SurfaceAt) {
    ImplicitSurfaceSet3 sset;

    auto box1 = std::make_shared<Box3>(BoundingBox3D({0, 0, 0}, {1, 2, 3}));
    auto box2 = std::make_shared<Box3>(BoundingBox3D({3, 4, 5}, {5, 6, 7}));
    sset.addExplicitSurface(box1);
    sset.addExplicitSurface(box2);

    auto implicitSurfaceAt0 =
        std::dynamic_pointer_cast<SurfaceToImplicit3>(sset.surfaceAt(0));
    auto implicitSurfaceAt1 =
        std::dynamic_pointer_cast<SurfaceToImplicit3>(sset.surfaceAt(1));

    EXPECT_EQ(std::dynamic_pointer_cast<Surface3>(box1),
              implicitSurfaceAt0->surface());
    EXPECT_EQ(std::dynamic_pointer_cast<Surface3>(box2),
              implicitSurfaceAt1->surface());
}

TEST(ImplicitSurfaceSet3, AddSurface) {
    ImplicitSurfaceSet3 sset;

    auto box1 = std::make_shared<Box3>(BoundingBox3D({0, 0, 0}, {1, 2, 3}));
    auto box2 = std::make_shared<Box3>(BoundingBox3D({3, 4, 5}, {5, 6, 7}));
    auto implicitBox = std::make_shared<SurfaceToImplicit3>(box2);

    sset.addExplicitSurface(box1);
    sset.addSurface(implicitBox);

    EXPECT_EQ(2u, sset.numberOfSurfaces());

    auto implicitSurfaceAt0 =
        std::dynamic_pointer_cast<SurfaceToImplicit3>(sset.surfaceAt(0));
    auto implicitSurfaceAt1 =
        std::dynamic_pointer_cast<SurfaceToImplicit3>(sset.surfaceAt(1));

    EXPECT_EQ(std::dynamic_pointer_cast<Surface3>(box1),
              implicitSurfaceAt0->surface());
    EXPECT_EQ(implicitBox, implicitSurfaceAt1);
}

TEST(ImplicitSurfaceSet3, ClosestPoint) {
    BoundingBox3D bbox(Vector3D(), Vector3D(1, 2, 3));

    Box3Ptr box = std::make_shared<Box3>(bbox);
    box->isNormalFlipped = true;

    ImplicitSurfaceSet3Ptr sset = std::make_shared<ImplicitSurfaceSet3>();
    Vector3D emptyPoint = sset->closestPoint({1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(kMaxD, emptyPoint.x);
    EXPECT_DOUBLE_EQ(kMaxD, emptyPoint.y);
    EXPECT_DOUBLE_EQ(kMaxD, emptyPoint.z);

    sset->addExplicitSurface(box);

    Vector3D pt(0.5, 2.5, -1.0);
    Vector3D boxPoint = box->closestPoint(pt);
    Vector3D setPoint = sset->closestPoint(pt);
    EXPECT_DOUBLE_EQ(boxPoint.x, setPoint.x);
    EXPECT_DOUBLE_EQ(boxPoint.y, setPoint.y);
    EXPECT_DOUBLE_EQ(boxPoint.z, setPoint.z);
}

TEST(ImplicitSurfaceSet3, ClosestDistance) {
    BoundingBox3D bbox(Vector3D(), Vector3D(1, 2, 3));

    Box3Ptr box = std::make_shared<Box3>(bbox);
    box->isNormalFlipped = true;

    ImplicitSurfaceSet3Ptr sset = std::make_shared<ImplicitSurfaceSet3>();
    sset->addExplicitSurface(box);

    Vector3D pt(0.5, 2.5, -1.0);
    double boxDist = box->closestDistance(pt);
    double setDist = sset->closestDistance(pt);
    EXPECT_DOUBLE_EQ(boxDist, setDist);
}

TEST(ImplicitSurfaceSet3, Intersects) {
    ImplicitSurfaceSet3 sset;
    auto box = std::make_shared<Box3>(BoundingBox3D({-1, 2, 3}, {5, 3, 7}));
    sset.addExplicitSurface(box);

    bool result0 = sset.intersects(
        Ray3D(Vector3D(1, 4, 5), Vector3D(-1, -1, -1).normalized()));
    EXPECT_TRUE(result0);

    bool result1 = sset.intersects(
        Ray3D(Vector3D(1, 2.5, 6), Vector3D(-1, -1, 1).normalized()));
    EXPECT_TRUE(result1);

    bool result2 = sset.intersects(
        Ray3D(Vector3D(1, 1, 2), Vector3D(-1, -1, -1).normalized()));
    EXPECT_FALSE(result2);
}

TEST(ImplicitSurfaceSet3, ClosestIntersection) {
    ImplicitSurfaceSet3 sset;
    auto box = std::make_shared<Box3>(BoundingBox3D({-1, 2, 3}, {5, 3, 7}));
    sset.addExplicitSurface(box);

    SurfaceRayIntersection3 result0 = sset.closestIntersection(
        Ray3D(Vector3D(1, 4, 5), Vector3D(-1, -1, -1).normalized()));
    EXPECT_TRUE(result0.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(3), result0.distance);
    EXPECT_EQ(Vector3D(0, 3, 4), result0.point);

    SurfaceRayIntersection3 result1 = sset.closestIntersection(
        Ray3D(Vector3D(1, 2.5, 6), Vector3D(-1, -1, 1).normalized()));
    EXPECT_TRUE(result1.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(0.75), result1.distance);
    EXPECT_EQ(Vector3D(0.5, 2, 6.5), result1.point);

    SurfaceRayIntersection3 result2 = sset.closestIntersection(
        Ray3D(Vector3D(1, 1, 2), Vector3D(-1, -1, -1).normalized()));
    EXPECT_FALSE(result2.isIntersecting);
}

TEST(ImplicitSurfaceSet3, BoundingBox) {
    ImplicitSurfaceSet3 sset;

    auto box1 = std::make_shared<Box3>(BoundingBox3D({0, -3, -1}, {1, 2, 4}));
    auto box2 = std::make_shared<Box3>(BoundingBox3D({3, 4, 2}, {5, 6, 9}));
    sset.addExplicitSurface(box1);
    sset.addExplicitSurface(box2);

    auto bbox = sset.boundingBox();
    EXPECT_DOUBLE_EQ(0.0, bbox.lowerCorner.x);
    EXPECT_DOUBLE_EQ(-3.0, bbox.lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, bbox.lowerCorner.z);
    EXPECT_DOUBLE_EQ(5.0, bbox.upperCorner.x);
    EXPECT_DOUBLE_EQ(6.0, bbox.upperCorner.y);
    EXPECT_DOUBLE_EQ(9.0, bbox.upperCorner.z);
}

TEST(ImplicitSurfaceSet3, SignedDistance) {
    BoundingBox3D bbox(Vector3D(1, 4, 3), Vector3D(5, 6, 9));

    Box3Ptr box = std::make_shared<Box3>(bbox);
    auto implicitBox = std::make_shared<SurfaceToImplicit3>(box);

    ImplicitSurfaceSet3Ptr sset = std::make_shared<ImplicitSurfaceSet3>();
    sset->addExplicitSurface(box);

    Vector3D pt(-1, 7, 8);
    double boxDist = implicitBox->signedDistance(pt);
    double setDist = sset->signedDistance(pt);
    EXPECT_DOUBLE_EQ(boxDist, setDist);
}

TEST(ImplicitSurfaceSet3, ClosestNormal) {
    BoundingBox3D bbox(Vector3D(), Vector3D(1, 2, 3));

    Box3Ptr box = std::make_shared<Box3>(bbox);
    box->isNormalFlipped = true;

    ImplicitSurfaceSet3Ptr sset = std::make_shared<ImplicitSurfaceSet3>();
    Vector3D emptyNormal = sset->closestNormal({1.0, 2.0, 3.0});
    // No expected value -- just see if it doesn't crash
    (void)emptyNormal;
    sset->addExplicitSurface(box);

    Vector3D pt(0.5, 2.5, -1.0);
    Vector3D boxNormal = box->closestNormal(pt);
    Vector3D setNormal = sset->closestNormal(pt);
    EXPECT_DOUBLE_EQ(boxNormal.x, setNormal.x);
    EXPECT_DOUBLE_EQ(boxNormal.y, setNormal.y);
    EXPECT_DOUBLE_EQ(boxNormal.z, setNormal.z);
}

TEST(ImplicitSurfaceSet3, MixedBoundTypes) {
    BoundingBox3D domain(Vector3D(), Vector3D(1, 2, 1));

    auto plane = Plane3::builder()
                     .withNormal({0, 1, 0})
                     .withPoint({0, 0.25 * domain.height(), 0})
                     .makeShared();

    auto sphere = Sphere3::builder()
                      .withCenter(domain.midPoint())
                      .withRadius(0.15 * domain.width())
                      .makeShared();

    auto surfaceSet = ImplicitSurfaceSet3::builder()
                          .withExplicitSurfaces({plane, sphere})
                          .makeShared();

    EXPECT_FALSE(surfaceSet->isBounded());

    auto cp = surfaceSet->closestPoint(Vector3D(0.5, 0.4, 0.5));
    Vector3D answer(0.5, 0.5, 0.5);

    EXPECT_VECTOR3_NEAR(answer, cp, 1e-9);
}

TEST(ImplicitSurfaceSet3, IsValidGeometry) {
    auto surfaceSet = ImplicitSurfaceSet3::builder().makeShared();

    EXPECT_FALSE(surfaceSet->isValidGeometry());

    auto box = std::make_shared<Box3>(BoundingBox3D({0, 0, 0}, {1, 2, 3}));
    auto surfaceSet2 = ImplicitSurfaceSet3::builder().makeShared();
    surfaceSet2->addExplicitSurface(box);

    EXPECT_TRUE(surfaceSet2->isValidGeometry());

    surfaceSet2->addSurface(surfaceSet);

    EXPECT_FALSE(surfaceSet2->isValidGeometry());
}

TEST(ImplicitSurfaceSet3, IsInside) {
    BoundingBox3D domain(Vector3D(), Vector3D(1, 2, 1));
    Vector3D offset(1, 2, 3);

    auto plane = Plane3::builder()
                     .withNormal({0, 1, 0})
                     .withPoint({0, 0.25 * domain.height(), 0})
                     .makeShared();

    auto sphere = Sphere3::builder()
                      .withCenter(domain.midPoint())
                      .withRadius(0.15 * domain.width())
                      .makeShared();

    auto surfaceSet = ImplicitSurfaceSet3::builder()
                          .withExplicitSurfaces({plane, sphere})
                          .withTransform(Transform3(offset, QuaternionD()))
                          .makeShared();

    EXPECT_TRUE(surfaceSet->isInside(Vector3D(0.5, 0.25, 0.5) + offset));
    EXPECT_TRUE(surfaceSet->isInside(Vector3D(0.5, 1.0, 0.5) + offset));
    EXPECT_FALSE(surfaceSet->isInside(Vector3D(0.5, 1.5, 0.5) + offset));
}

TEST(ImplicitSurfaceSet3, UpdateQueryEngine) {
    auto sphere = Sphere3::builder()
                      .withCenter({-1.0, 1.0, 2.0})
                      .withRadius(0.5)
                      .makeShared();

    auto surfaceSet =
        ImplicitSurfaceSet3::builder()
            .withExplicitSurfaces({sphere})
            .withTransform(Transform3({1.0, 2.0, -1.0}, QuaternionD()))
            .makeShared();

    auto bbox1 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox3D({-0.5, 2.5, 0.5}, {0.5, 3.5, 1.5}),
                            bbox1);

    surfaceSet->transform = Transform3({3.0, -4.0, 7.0}, QuaternionD());
    surfaceSet->updateQueryEngine();
    auto bbox2 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox3D({1.5, -3.5, 4.5}, {2.5, -2.5, 5.5}),
                            bbox2);

    sphere->transform = Transform3({-6.0, 9.0, 2.0}, QuaternionD());
    surfaceSet->updateQueryEngine();
    auto bbox3 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox3D({-4.5, 5.5, 10.5}, {-3.5, 6.5, 11.5}),
                            bbox3);

    // Plane is unbounded. Total bbox should ignore it.
    auto plane = Plane3::builder().withNormal({1.0, 0.0, 0.0}).makeShared();
    surfaceSet->addExplicitSurface(plane);
    surfaceSet->updateQueryEngine();
    auto bbox4 = surfaceSet->boundingBox();
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox3D({-4.5, 5.5, 10.5}, {-3.5, 6.5, 11.5}),
                            bbox4);
}
