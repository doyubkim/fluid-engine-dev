// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/box2.h>
#include <jet/plane2.h>
#include <jet/surface_set2.h>
#include <jet/surface_to_implicit2.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace jet;

namespace {

class MockSurface2 final : public Surface2 {
 public:
    MockSurface2() = default;

    ~MockSurface2() = default;

    MOCK_METHOD0(updateQueryEngine, void());

 protected:
    MOCK_CONST_METHOD1(closestPointLocal, Vector2D(const Vector2D&));
    MOCK_CONST_METHOD0(boundingBoxLocal, BoundingBox2D());
    MOCK_CONST_METHOD1(closestIntersectionLocal,
                       SurfaceRayIntersection2(const Ray2D&));
    MOCK_CONST_METHOD1(closestNormalLocal, Vector2D(const Vector2D&));
};

}  // namespace

TEST(SurfaceToImplicit2, Constructor) {
    auto box = std::make_shared<Box2>(BoundingBox2D({0, 0}, {1, 2}));

    SurfaceToImplicit2 s2i(box);
    EXPECT_EQ(box, s2i.surface());

    s2i.isNormalFlipped = true;
    SurfaceToImplicit2 s2i2(s2i);
    EXPECT_EQ(box, s2i2.surface());
    EXPECT_TRUE(s2i2.isNormalFlipped);
}

TEST(SurfaceToImplicit2, UpdateQueryEngine) {
    auto mockSurface2 = std::make_shared<MockSurface2>();
    SurfaceToImplicit2 s2i(mockSurface2);

    EXPECT_CALL(*mockSurface2, updateQueryEngine()).Times(1);

    s2i.updateQueryEngine();
}

TEST(SurfaceToImplicit2, ClosestPoint) {
    BoundingBox2D bbox(Vector2D(), Vector2D(1, 2));

    Box2Ptr box = std::make_shared<Box2>(bbox);

    SurfaceToImplicit2 s2i(box);

    Vector2D pt(0.5, 2.5);
    Vector2D boxPoint = box->closestPoint(pt);
    Vector2D s2iPoint = s2i.closestPoint(pt);
    EXPECT_DOUBLE_EQ(boxPoint.x, s2iPoint.x);
    EXPECT_DOUBLE_EQ(boxPoint.y, s2iPoint.y);
}

TEST(SurfaceToImplicit2, ClosestDistance) {
    BoundingBox2D bbox(Vector2D(), Vector2D(1, 2));

    Box2Ptr box = std::make_shared<Box2>(bbox);

    SurfaceToImplicit2 s2i(box);

    Vector2D pt(0.5, 2.5);
    double boxDist = box->closestDistance(pt);
    double s2iDist = s2i.closestDistance(pt);
    EXPECT_DOUBLE_EQ(boxDist, s2iDist);
}

TEST(SurfaceToImplicit2, Intersects) {
    auto box = std::make_shared<Box2>(BoundingBox2D({-1, 2}, {5, 3}));
    SurfaceToImplicit2 s2i(box);

    bool result0 =
        s2i.intersects(Ray2D(Vector2D(1, 4), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result0);

    bool result1 =
        s2i.intersects(Ray2D(Vector2D(1, 2.5), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result1);

    bool result2 =
        s2i.intersects(Ray2D(Vector2D(1, 1), Vector2D(-1, -1).normalized()));
    EXPECT_FALSE(result2);
}

TEST(SurfaceToImplicit2, ClosestIntersection) {
    auto box = std::make_shared<Box2>(BoundingBox2D({-1, 2}, {5, 3}));
    SurfaceToImplicit2 s2i(box);

    SurfaceRayIntersection2 result0 = s2i.closestIntersection(
        Ray2D(Vector2D(1, 4), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result0.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(2), result0.distance);
    EXPECT_EQ(Vector2D(0, 3), result0.point);

    SurfaceRayIntersection2 result1 = s2i.closestIntersection(
        Ray2D(Vector2D(1, 2.5), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result1.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(0.5), result1.distance);
    EXPECT_EQ(Vector2D(0.5, 2), result1.point);

    SurfaceRayIntersection2 result2 = s2i.closestIntersection(
        Ray2D(Vector2D(1, 1), Vector2D(-1, -1).normalized()));
    EXPECT_FALSE(result2.isIntersecting);
}

TEST(SurfaceToImplicit2, BoundingBox) {
    auto box = std::make_shared<Box2>(BoundingBox2D({-1, 2}, {5, 3}));
    SurfaceToImplicit2 s2i(box);

    auto bbox = s2i.boundingBox();
    EXPECT_DOUBLE_EQ(-1.0, bbox.lowerCorner.x);
    EXPECT_DOUBLE_EQ(2.0, bbox.lowerCorner.y);
    EXPECT_DOUBLE_EQ(5.0, bbox.upperCorner.x);
    EXPECT_DOUBLE_EQ(3.0, bbox.upperCorner.y);
}

TEST(SurfaceToImplicit2, SignedDistance) {
    BoundingBox2D bbox(Vector2D(1, 4), Vector2D(5, 6));

    Box2Ptr box = std::make_shared<Box2>(bbox);
    SurfaceToImplicit2 s2i(box);

    Vector2D pt(-1, 7);
    double boxDist = box->closestDistance(pt);
    double s2iDist = s2i.signedDistance(pt);
    EXPECT_DOUBLE_EQ(boxDist, s2iDist);

    s2i.isNormalFlipped = true;
    s2iDist = s2i.signedDistance(pt);
    EXPECT_DOUBLE_EQ(-boxDist, s2iDist);
}

TEST(SurfaceToImplicit2, ClosestNormal) {
    BoundingBox2D bbox(Vector2D(), Vector2D(1, 2));

    Box2Ptr box = std::make_shared<Box2>(bbox);
    box->isNormalFlipped = true;

    SurfaceToImplicit2 s2i(box);

    Vector2D pt(0.5, 2.5);
    Vector2D boxNormal = box->closestNormal(pt);
    Vector2D s2iNormal = s2i.closestNormal(pt);
    EXPECT_DOUBLE_EQ(boxNormal.x, s2iNormal.x);
    EXPECT_DOUBLE_EQ(boxNormal.y, s2iNormal.y);
}

TEST(SurfaceToImplicit2, IsBounded) {
    Plane2Ptr plane =
        Plane2::builder().withPoint({0, 0}).withNormal({0, 1}).makeShared();
    SurfaceToImplicit2Ptr s2i =
        SurfaceToImplicit2::builder().withSurface(plane).makeShared();
    EXPECT_FALSE(s2i->isBounded());
}

TEST(SurfaceToImplicit2, IsValidGeometry) {
    SurfaceSet2Ptr sset = SurfaceSet2::builder().makeShared();
    SurfaceToImplicit2Ptr s2i =
        SurfaceToImplicit2::builder().withSurface(sset).makeShared();
    EXPECT_FALSE(s2i->isValidGeometry());
}

TEST(SurfaceToImplicit2, IsInside) {
    Plane2Ptr plane = Plane2::builder()
                          .withPoint({0, 0})
                          .withNormal({0, 1})
                          .withTranslation({0, -1})
                          .makeShared();
    SurfaceToImplicit2Ptr s2i =
        SurfaceToImplicit2::builder().withSurface(plane).makeShared();
    EXPECT_FALSE(s2i->isInside({0, -0.5}));
    EXPECT_TRUE(s2i->isInside({0, -1.5}));
}
