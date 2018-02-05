// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/quadtree.h>

using namespace jet;

TEST(Quadtree, Constructors) {
    Quadtree<Vector2D> quadtree;
    EXPECT_EQ(quadtree.begin(), quadtree.end());
}

TEST(Quadtree, Nearest) {
    Quadtree<Vector2D> quadtree;

    auto overlapsFunc = [](const Vector2D& pt, const BoundingBox2D& bbox) {
        return bbox.contains(pt);
    };

    auto distanceFunc = [](const Vector2D& a, const Vector2D& b) {
        return a.distanceTo(b);
    };

    // Single point
    quadtree.build({Vector2D(0.2, 0.7)}, BoundingBox2D({0, 0}, {0.9, 1.0}),
                   overlapsFunc, 3);

    EXPECT_EQ(3u, quadtree.maxDepth());
    EXPECT_VECTOR2_EQ(Vector2D(0, 0), quadtree.boundingBox().lowerCorner);
    EXPECT_VECTOR2_EQ(Vector2D(1, 1), quadtree.boundingBox().upperCorner);
    EXPECT_EQ(9u, quadtree.numberOfNodes());

    size_t child = quadtree.childIndex(0, 2);
    EXPECT_EQ(3u, child);

    child = quadtree.childIndex(child, 0);
    EXPECT_EQ(5u, child);

    size_t theNonEmptyLeafNode = child + 0;
    for (size_t i = 0; i < 9; ++i) {
        if (i == theNonEmptyLeafNode) {
            EXPECT_EQ(1u, quadtree.itemsAtNode(i).size());
        } else {
            EXPECT_EQ(0u, quadtree.itemsAtNode(i).size());
        }
    }

    // Many points
    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<Vector2D> points(getSamplePoints2(),
                                 getSamplePoints2() + numSamples);

    quadtree.build(points, BoundingBox2D({0, 0}, {1, 1}), overlapsFunc, 5);

    Vector2D testPt(0.5, 0.5);
    auto nearest = quadtree.nearest(testPt, distanceFunc);
    ptrdiff_t answerIdx = 0;
    double bestDist = testPt.distanceTo(points[answerIdx]);
    for (size_t i = 1; i < numSamples; ++i) {
        double dist = testPt.distanceTo(getSamplePoints2()[i]);
        if (dist < bestDist) {
            bestDist = dist;
            answerIdx = i;
        }
    }

    EXPECT_EQ(answerIdx, nearest.item - &quadtree.item(0));
}

TEST(Quadtree, BBoxIntersects) {
    Quadtree<Vector2D> quadtree;

    auto overlapsFunc = [](const Vector2D& pt, const BoundingBox2D& bbox) {
        return bbox.contains(pt);
    };

    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<Vector2D> points(getSamplePoints2(),
                                 getSamplePoints2() + numSamples);

    quadtree.build(points, BoundingBox2D({0, 0}, {1, 1}), overlapsFunc, 5);

    BoundingBox2D testBox({0.25, 0.15}, {0.5, 0.6});
    bool hasOverlaps = false;
    for (size_t i = 0; i < numSamples; ++i) {
        hasOverlaps |= overlapsFunc(getSamplePoints2()[i], testBox);
    }

    EXPECT_EQ(hasOverlaps, quadtree.intersects(testBox, overlapsFunc));

    BoundingBox2D testBox2({0.2, 0.2}, {0.6, 0.5});
    hasOverlaps = false;
    for (size_t i = 0; i < numSamples; ++i) {
        hasOverlaps |= overlapsFunc(getSamplePoints2()[i], testBox2);
    }

    EXPECT_EQ(hasOverlaps, quadtree.intersects(testBox2, overlapsFunc));
}

TEST(Quadtree, ForEachOverlappingItems) {
    Quadtree<Vector2D> quadtree;

    auto overlapsFunc = [](const Vector2D& pt, const BoundingBox2D& bbox) {
        return bbox.contains(pt);
    };

    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<Vector2D> points(getSamplePoints2(),
                                 getSamplePoints2() + numSamples);

    quadtree.build(points, BoundingBox2D({0, 0}, {1, 1}), overlapsFunc, 5);

    BoundingBox2D testBox({0.2, 0.2}, {0.6, 0.5});
    size_t numOverlaps = 0;
    for (size_t i = 0; i < numSamples; ++i) {
        numOverlaps += overlapsFunc(getSamplePoints2()[i], testBox);
    }

    size_t measured = 0;
    quadtree.forEachIntersectingItem(testBox, overlapsFunc,
                                     [&](const Vector2D& pt) {
                                         EXPECT_TRUE(overlapsFunc(pt, testBox));
                                         ++measured;
                                     });

    EXPECT_EQ(numOverlaps, measured);
}

TEST(Quadtree, RayIntersects) {
    Quadtree<BoundingBox2D> quadtree;

    auto overlapsFunc = [](const BoundingBox2D& a, const BoundingBox2D& bbox) {
        return bbox.overlaps(a);
    };

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

    quadtree.build(items, BoundingBox2D({0, 0}, {1, 1}), overlapsFunc, 5);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray2D ray(getSamplePoints2()[i + numSamples / 2],
                  getSampleDirs2()[i + numSamples / 2]);
        // ad-hoc search
        bool ansInts = false;
        for (size_t j = 0; j < numSamples / 2; ++j) {
            if (intersectsFunc(items[j], ray)) {
                ansInts = true;
                break;
            }
        }

        // quadtree search
        bool octInts = quadtree.intersects(ray, intersectsFunc);

        EXPECT_EQ(ansInts, octInts);
    }
}

TEST(Quadtree, ClosestIntersection) {
    Quadtree<BoundingBox2D> quadtree;

    auto overlapsFunc = [](const BoundingBox2D& a, const BoundingBox2D& bbox) {
        return bbox.overlaps(a);
    };

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

    quadtree.build(items, BoundingBox2D({0, 0}, {1, 1}), overlapsFunc, 5);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray2D ray(getSamplePoints2()[i + numSamples / 2],
                  getSampleDirs2()[i + numSamples / 2]);
        // ad-hoc search
        ClosestIntersectionQueryResult2<BoundingBox2D> ansInts;
        for (size_t j = 0; j < numSamples / 2; ++j) {
            double dist = intersectsFunc(items[j], ray);
            if (dist < ansInts.distance) {
                ansInts.distance = dist;
                ansInts.item = &quadtree.item(j);
            }
        }

        // quadtree search
        auto octInts = quadtree.closestIntersection(ray, intersectsFunc);

        EXPECT_DOUBLE_EQ(ansInts.distance, octInts.distance);
        EXPECT_EQ(ansInts.item, octInts.item);
    }
}
