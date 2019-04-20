// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/box3.h>
#include <jet/plane3.h>
#include <jet/surface_set3.h>
#include <jet/surface_to_implicit3.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace jet;

namespace {

class MockSurface3 final : public Surface3 {
 public:
    MockSurface3() = default;

    ~MockSurface3() = default;

    MOCK_METHOD0(updateQueryEngine, void());

 protected:
    MOCK_CONST_METHOD1(closestPointLocal, Vector3D(const Vector3D&));
    MOCK_CONST_METHOD0(boundingBoxLocal, BoundingBox3D());
    MOCK_CONST_METHOD1(closestIntersectionLocal,
                       SurfaceRayIntersection3(const Ray3D&));
    MOCK_CONST_METHOD1(closestNormalLocal, Vector3D(const Vector3D&));
};

}  // namespace

TEST(SurfaceToImplicit3, Constructor) {
    auto box = std::make_shared<Box3>(BoundingBox3D({0, 0, 0}, {1, 2, 3}));

    SurfaceToImplicit3 s2i(box);
    EXPECT_EQ(box, s2i.surface());

    s2i.isNormalFlipped = true;
    SurfaceToImplicit3 s2i2(s2i);
    EXPECT_EQ(box, s2i2.surface());
    EXPECT_TRUE(s2i2.isNormalFlipped);
}

TEST(SurfaceToImplicit3, UpdateQueryEngine) {
    auto mockSurface3 = std::make_shared<MockSurface3>();
    SurfaceToImplicit3 s2i(mockSurface3);

    EXPECT_CALL(*mockSurface3, updateQueryEngine()).Times(1);

    s2i.updateQueryEngine();
}

TEST(SurfaceToImplicit3, ClosestPoint) {
    BoundingBox3D bbox(Vector3D(), Vector3D(1, 2, 3));

    Box3Ptr box = std::make_shared<Box3>(bbox);

    SurfaceToImplicit3 s2i(box);

    Vector3D pt(0.5, 2.5, -1.0);
    Vector3D boxPoint = box->closestPoint(pt);
    Vector3D s2iPoint = s2i.closestPoint(pt);
    EXPECT_DOUBLE_EQ(boxPoint.x, s2iPoint.x);
    EXPECT_DOUBLE_EQ(boxPoint.y, s2iPoint.y);
    EXPECT_DOUBLE_EQ(boxPoint.z, s2iPoint.z);
}

TEST(SurfaceToImplicit3, ClosestDistance) {
    BoundingBox3D bbox(Vector3D(), Vector3D(1, 2, 3));

    Box3Ptr box = std::make_shared<Box3>(bbox);

    SurfaceToImplicit3 s2i(box);

    Vector3D pt(0.5, 2.5, -1.0);
    double boxDist = box->closestDistance(pt);
    double s2iDist = s2i.closestDistance(pt);
    EXPECT_DOUBLE_EQ(boxDist, s2iDist);
}

TEST(SurfaceToImplicit3, Intersects) {
    auto box = std::make_shared<Box3>(BoundingBox3D({-1, 2, 3}, {5, 3, 7}));
    SurfaceToImplicit3 s2i(box);

    bool result0 = s2i.intersects(
        Ray3D(Vector3D(1, 4, 5), Vector3D(-1, -1, -1).normalized()));
    EXPECT_TRUE(result0);

    bool result1 = s2i.intersects(
        Ray3D(Vector3D(1, 2.5, 6), Vector3D(-1, -1, 1).normalized()));
    EXPECT_TRUE(result1);

    bool result2 = s2i.intersects(
        Ray3D(Vector3D(1, 1, 2), Vector3D(-1, -1, -1).normalized()));
    EXPECT_FALSE(result2);
}

TEST(SurfaceToImplicit3, ClosestIntersection) {
    auto box = std::make_shared<Box3>(BoundingBox3D({-1, 2, 3}, {5, 3, 7}));
    SurfaceToImplicit3 s2i(box);

    SurfaceRayIntersection3 result0 = s2i.closestIntersection(
        Ray3D(Vector3D(1, 4, 5), Vector3D(-1, -1, -1).normalized()));
    EXPECT_TRUE(result0.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(3), result0.distance);
    EXPECT_EQ(Vector3D(0, 3, 4), result0.point);

    SurfaceRayIntersection3 result1 = s2i.closestIntersection(
        Ray3D(Vector3D(1, 2.5, 6), Vector3D(-1, -1, 1).normalized()));
    EXPECT_TRUE(result1.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(0.75), result1.distance);
    EXPECT_EQ(Vector3D(0.5, 2, 6.5), result1.point);

    SurfaceRayIntersection3 result2 = s2i.closestIntersection(
        Ray3D(Vector3D(1, 1, 2), Vector3D(-1, -1, -1).normalized()));
    EXPECT_FALSE(result2.isIntersecting);
}

TEST(SurfaceToImplicit3, BoundingBox) {
    auto box = std::make_shared<Box3>(BoundingBox3D({0, -3, -1}, {1, 2, 4}));
    SurfaceToImplicit3 s2i(box);

    auto bbox = s2i.boundingBox();
    EXPECT_DOUBLE_EQ(0.0, bbox.lowerCorner.x);
    EXPECT_DOUBLE_EQ(-3.0, bbox.lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, bbox.lowerCorner.z);
    EXPECT_DOUBLE_EQ(1.0, bbox.upperCorner.x);
    EXPECT_DOUBLE_EQ(2.0, bbox.upperCorner.y);
    EXPECT_DOUBLE_EQ(4.0, bbox.upperCorner.z);
}

TEST(SurfaceToImplicit3, SignedDistance) {
    BoundingBox3D bbox(Vector3D(1, 4, 3), Vector3D(5, 6, 9));

    Box3Ptr box = std::make_shared<Box3>(bbox);
    SurfaceToImplicit3 s2i(box);

    Vector3D pt(-1, 7, 8);
    double boxDist = box->closestDistance(pt);
    double s2iDist = s2i.signedDistance(pt);
    EXPECT_DOUBLE_EQ(boxDist, s2iDist);

    s2i.isNormalFlipped = true;
    s2iDist = s2i.signedDistance(pt);
    EXPECT_DOUBLE_EQ(-boxDist, s2iDist);
}

TEST(SurfaceToImplicit3, ClosestNormal) {
    BoundingBox3D bbox(Vector3D(), Vector3D(1, 2, 3));

    Box3Ptr box = std::make_shared<Box3>(bbox);
    box->isNormalFlipped = true;

    SurfaceToImplicit3 s2i(box);

    Vector3D pt(0.5, 2.5, -1.0);
    Vector3D boxNormal = box->closestNormal(pt);
    Vector3D s2iNormal = s2i.closestNormal(pt);
    EXPECT_DOUBLE_EQ(boxNormal.x, s2iNormal.x);
    EXPECT_DOUBLE_EQ(boxNormal.y, s2iNormal.y);
    EXPECT_DOUBLE_EQ(boxNormal.z, s2iNormal.z);
}

TEST(SurfaceToImplicit3, IsBounded) {
    Plane3Ptr plane = Plane3::builder()
                          .withPoint({0, 0, 0})
                          .withNormal({0, 1, 0})
                          .makeShared();
    SurfaceToImplicit3Ptr s2i =
        SurfaceToImplicit3::builder().withSurface(plane).makeShared();
    EXPECT_FALSE(s2i->isBounded());
}

TEST(SurfaceToImplicit3, IsValidGeometry) {
    SurfaceSet3Ptr sset = SurfaceSet3::builder().makeShared();
    SurfaceToImplicit3Ptr s2i =
        SurfaceToImplicit3::builder().withSurface(sset).makeShared();
    EXPECT_FALSE(s2i->isValidGeometry());
}

TEST(SurfaceToImplicit3, IsInside) {
    Plane3Ptr plane = Plane3::builder()
                          .withPoint({0, 0, 0})
                          .withNormal({0, 1, 0})
                          .withTranslation({0, -1, 0})
                          .makeShared();
    SurfaceToImplicit3Ptr s2i =
        SurfaceToImplicit3::builder().withSurface(plane).makeShared();
    EXPECT_FALSE(s2i->isInside({0, -0.5, 0}));
    EXPECT_TRUE(s2i->isInside({0, -1.5, 0}));
}
