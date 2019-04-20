// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/bvh2.h>

using namespace jet;

TEST(Bvh2, Constructors) {
    Bvh2<Vector2D> bvh;
    EXPECT_EQ(bvh.begin(), bvh.end());
}

TEST(Bvh2, BasicGetters) {
    Bvh2<Vector2D> bvh;

    std::vector<Vector2D> points{Vector2D(0, 0), Vector2D(1, 1)};
    std::vector<BoundingBox2D> bounds(points.size());
    size_t i = 0;
    BoundingBox2D rootBounds;
    std::generate(bounds.begin(), bounds.end(), [&]() {
        auto c = points[i++];
        BoundingBox2D box(c, c);
        box.expand(0.1);
        rootBounds.merge(box);
        return box;
    });

    bvh.build(points, bounds);

    EXPECT_EQ(2u, bvh.numberOfItems());
    EXPECT_VECTOR2_EQ(points[0], bvh.item(0));
    EXPECT_VECTOR2_EQ(points[1], bvh.item(1));
    EXPECT_EQ(3u, bvh.numberOfNodes());
    EXPECT_EQ(1u, bvh.children(0).first);
    EXPECT_EQ(2u, bvh.children(0).second);
    EXPECT_FALSE(bvh.isLeaf(0));
    EXPECT_TRUE(bvh.isLeaf(1));
    EXPECT_TRUE(bvh.isLeaf(2));
    EXPECT_BOUNDING_BOX2_EQ(rootBounds, bvh.nodeBound(0));
    EXPECT_BOUNDING_BOX2_EQ(bounds[0], bvh.nodeBound(1));
    EXPECT_BOUNDING_BOX2_EQ(bounds[1], bvh.nodeBound(2));
    EXPECT_EQ(bvh.end(), bvh.itemOfNode(0));
    EXPECT_EQ(bvh.begin(), bvh.itemOfNode(1));
    EXPECT_EQ(bvh.begin() + 1, bvh.itemOfNode(2));
}

TEST(Bvh2, Nearest) {
    Bvh2<Vector2D> bvh;

    auto distanceFunc = [](const Vector2D& a, const Vector2D& b) {
        return a.distanceTo(b);
    };

    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<Vector2D> points(getSamplePoints2(),
                                 getSamplePoints2() + numSamples);

    std::vector<BoundingBox2D> bounds(points.size());
    size_t i = 0;
    std::generate(bounds.begin(), bounds.end(), [&]() {
        auto c = points[i++];
        BoundingBox2D box(c, c);
        box.expand(0.1);
        return box;
    });

    bvh.build(points, bounds);

    Vector2D testPt(0.5, 0.5);
    auto nearest = bvh.nearest(testPt, distanceFunc);
    ptrdiff_t answerIdx = 0;
    double bestDist = testPt.distanceTo(points[answerIdx]);
    for (i = 1; i < numSamples; ++i) {
        double dist = testPt.distanceTo(getSamplePoints2()[i]);
        if (dist < bestDist) {
            bestDist = dist;
            answerIdx = i;
        }
    }

    EXPECT_EQ(answerIdx, nearest.item - &bvh.item(0));
}

TEST(Bvh2, BBoxIntersects) {
    Bvh2<Vector2D> bvh;

    auto overlapsFunc = [](const Vector2D& pt, const BoundingBox2D& bbox) {
        BoundingBox2D box(pt, pt);
        box.expand(0.1);
        return bbox.overlaps(box);
    };

    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<Vector2D> points(getSamplePoints2(),
                                 getSamplePoints2() + numSamples);

    std::vector<BoundingBox2D> bounds(points.size());
    size_t i = 0;
    std::generate(bounds.begin(), bounds.end(), [&]() {
        auto c = points[i++];
        BoundingBox2D box(c, c);
        box.expand(0.1);
        return box;
    });

    bvh.build(points, bounds);

    BoundingBox2D testBox({0.25, 0.15}, {0.5, 0.6});
    bool hasOverlaps = false;
    for (i = 0; i < numSamples; ++i) {
        hasOverlaps |= overlapsFunc(getSamplePoints2()[i], testBox);
    }

    EXPECT_EQ(hasOverlaps, bvh.intersects(testBox, overlapsFunc));

    BoundingBox2D testBox2({0.2, 0.2}, {0.6, 0.5});
    hasOverlaps = false;
    for (i = 0; i < numSamples; ++i) {
        hasOverlaps |= overlapsFunc(getSamplePoints2()[i], testBox2);
    }

    EXPECT_EQ(hasOverlaps, bvh.intersects(testBox2, overlapsFunc));
}

TEST(Bvh2, RayIntersects) {
    Bvh2<BoundingBox2D> bvh;

    auto intersectsFunc = [](const BoundingBox2D& a, const Ray2D& ray) {
        return a.intersects(ray);
    };

    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<BoundingBox2D> items(numSamples / 2);
    size_t i = 0;
    std::generate(items.begin(), items.end(), [&]() {
        auto c = getSamplePoints2()[i++];
        BoundingBox2D box(c, c);
        box.expand(0.1);
        return box;
    });

    bvh.build(items, items);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray2D ray(getSampleDirs2()[i + numSamples / 2],
                  getSampleDirs2()[i + numSamples / 2]);
        // ad-hoc search
        bool ansInts = false;
        for (size_t j = 0; j < numSamples / 2; ++j) {
            if (intersectsFunc(items[j], ray)) {
                ansInts = true;
                break;
            }
        }

        // bvh search
        bool octInts = bvh.intersects(ray, intersectsFunc);

        EXPECT_EQ(ansInts, octInts);
    }
}

TEST(Bvh2, ClosestIntersection) {
    Bvh2<BoundingBox2D> bvh;

    auto intersectsFunc = [](const BoundingBox2D& a, const Ray2D& ray) {
        auto bboxResult = a.closestIntersection(ray);
        if (bboxResult.isIntersecting) {
            return bboxResult.tNear;
        } else {
            return kMaxD;
        }
    };

    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<BoundingBox2D> items(numSamples / 2);
    size_t i = 0;
    std::generate(items.begin(), items.end(), [&]() {
        auto c = getSamplePoints2()[i++];
        BoundingBox2D box(c, c);
        box.expand(0.1);
        return box;
    });

    bvh.build(items, items);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray2D ray(getSamplePoints2()[i + numSamples / 2],
                  getSampleDirs2()[i + numSamples / 2]);
        // ad-hoc search
        ClosestIntersectionQueryResult2<BoundingBox2D> ansInts;
        for (size_t j = 0; j < numSamples / 2; ++j) {
            double dist = intersectsFunc(items[j], ray);
            if (dist < ansInts.distance) {
                ansInts.distance = dist;
                ansInts.item = &bvh.item(j);
            }
        }

        // bvh search
        auto bvhInts = bvh.closestIntersection(ray, intersectsFunc);

        EXPECT_DOUBLE_EQ(ansInts.distance, bvhInts.distance);
        EXPECT_EQ(ansInts.item, bvhInts.item);
    }
}

TEST(Bvh2, ForEachOverlappingItems) {
    Bvh2<Vector2D> bvh;

    auto overlapsFunc = [](const Vector2D& pt, const BoundingBox2D& bbox) {
        return bbox.contains(pt);
    };

    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<Vector2D> points(getSamplePoints2(),
                                 getSamplePoints2() + numSamples);

    std::vector<BoundingBox2D> bounds(points.size());
    size_t i = 0;
    std::generate(bounds.begin(), bounds.end(), [&]() {
        auto c = points[i++];
        BoundingBox2D box(c, c);
        box.expand(0.1);
        return box;
    });

    bvh.build(points, bounds);

    BoundingBox2D testBox({0.2, 0.2}, {0.6, 0.5});
    size_t numOverlaps = 0;
    for (i = 0; i < numSamples; ++i) {
        numOverlaps += overlapsFunc(getSamplePoints2()[i], testBox);
    }

    size_t measured = 0;
    bvh.forEachIntersectingItem(testBox, overlapsFunc, [&](const Vector2D& pt) {
        EXPECT_TRUE(overlapsFunc(pt, testBox));
        ++measured;
    });

    EXPECT_EQ(numOverlaps, measured);
}
