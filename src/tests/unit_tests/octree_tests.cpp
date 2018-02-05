// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/octree.h>

using namespace jet;

TEST(Octree, Constructors) {
    Octree<Vector3D> octree;
    EXPECT_EQ(octree.begin(), octree.end());
}

TEST(Octree, Nearest) {
    Octree<Vector3D> octree;

    auto overlapsFunc = [](const Vector3D& pt, const BoundingBox3D& bbox) {
        return bbox.contains(pt);
    };

    auto distanceFunc = [](const Vector3D& a, const Vector3D& b) {
        return a.distanceTo(b);
    };

    // Single point
    octree.build({Vector3D(0.2, 0.7, 0.3)},
                 BoundingBox3D({0, 0, 0}, {0.9, 0.8, 1}), overlapsFunc, 3);

    EXPECT_EQ(3u, octree.maxDepth());
    EXPECT_VECTOR3_EQ(Vector3D(0, 0, 0), octree.boundingBox().lowerCorner);
    EXPECT_VECTOR3_EQ(Vector3D(1, 1, 1), octree.boundingBox().upperCorner);
    EXPECT_EQ(17u, octree.numberOfNodes());

    size_t child = octree.childIndex(0, 2);
    EXPECT_EQ(3u, child);

    child = octree.childIndex(child, 0);
    EXPECT_EQ(9u, child);

    size_t theNonEmptyLeafNode = child + 4;
    for (size_t i = 0; i < 17; ++i) {
        if (i == theNonEmptyLeafNode) {
            EXPECT_EQ(1u, octree.itemsAtNode(i).size());
        } else {
            EXPECT_EQ(0u, octree.itemsAtNode(i).size());
        }
    }

    // Many points
    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<Vector3D> points(getSamplePoints3(), getSamplePoints3() + numSamples);

    octree.build(points, BoundingBox3D({0, 0, 0}, {1, 1, 1}), overlapsFunc, 5);

    Vector3D testPt(0.5, 0.5, 0.5);
    auto nearest = octree.nearest(testPt, distanceFunc);
    ptrdiff_t answerIdx = 0;
    double bestDist = testPt.distanceTo(points[answerIdx]);
    for (size_t i = 1; i < numSamples; ++i) {
        double dist = testPt.distanceTo(getSamplePoints3()[i]);
        if (dist < bestDist) {
            bestDist = dist;
            answerIdx = i;
        }
    }

    EXPECT_EQ(answerIdx, nearest.item - &octree.item(0));
}

TEST(Octree, BBoxIntersects) {
    Octree<Vector3D> octree;

    auto overlapsFunc = [](const Vector3D& pt, const BoundingBox3D& bbox) {
        return bbox.contains(pt);
    };

    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<Vector3D> points(getSamplePoints3(), getSamplePoints3() + numSamples);

    octree.build(points, BoundingBox3D({0, 0, 0}, {1, 1, 1}), overlapsFunc, 5);

    BoundingBox3D testBox({0.25, 0.15, 0.3}, {0.5, 0.6, 0.4});
    bool hasOverlaps = false;
    for (size_t i = 0; i < numSamples; ++i) {
        hasOverlaps |= overlapsFunc(getSamplePoints3()[i], testBox);
    }

    EXPECT_EQ(hasOverlaps, octree.intersects(testBox, overlapsFunc));

    BoundingBox3D testBox2({0.3, 0.2, 0.1}, {0.6, 0.5, 0.4});
    hasOverlaps = false;
    for (size_t i = 0; i < numSamples; ++i) {
        hasOverlaps |= overlapsFunc(getSamplePoints3()[i], testBox2);
    }

    EXPECT_EQ(hasOverlaps, octree.intersects(testBox2, overlapsFunc));
}

TEST(Octree, ForEachOverlappingItems) {
    Octree<Vector3D> octree;

    auto overlapsFunc = [](const Vector3D& pt, const BoundingBox3D& bbox) {
        return bbox.contains(pt);
    };

    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<Vector3D> points(getSamplePoints3(), getSamplePoints3() + numSamples);

    octree.build(points, BoundingBox3D({0, 0, 0}, {1, 1, 1}), overlapsFunc, 5);

    BoundingBox3D testBox({0.3, 0.2, 0.1}, {0.6, 0.5, 0.4});
    size_t numOverlaps = 0;
    for (size_t i = 0; i < numSamples; ++i) {
        numOverlaps += overlapsFunc(getSamplePoints3()[i], testBox);
    }

    size_t measured = 0;
    octree.forEachIntersectingItem(testBox, overlapsFunc,
                                   [&](const Vector3D& pt) {
                                       EXPECT_TRUE(overlapsFunc(pt, testBox));
                                       ++measured;
                                   });

    EXPECT_EQ(numOverlaps, measured);
}

TEST(Octree, RayIntersects) {
    Octree<BoundingBox3D> octree;

    auto overlapsFunc = [](const BoundingBox3D& a, const BoundingBox3D& bbox) {
        return bbox.overlaps(a);
    };

    auto intersectsFunc = [](const BoundingBox3D& a, const Ray3D& ray) {
        return a.intersects(ray);
    };

    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<BoundingBox3D> items(numSamples / 2);
    size_t i = 0;
    std::generate(items.begin(), items.end(), [&]() {
        auto c = getSamplePoints3()[i++];
        BoundingBox3D box(c, c);
        box.expand(0.1);
        return box;
    });

    octree.build(items, BoundingBox3D({0, 0, 0}, {1, 1, 1}), overlapsFunc, 5);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray3D ray(getSamplePoints3()[i + numSamples / 2],
                  getSampleDirs3()[i + numSamples / 2]);
        // ad-hoc search
        bool ansInts = false;
        for (size_t j = 0; j < numSamples / 2; ++j) {
            if (intersectsFunc(items[j], ray)) {
                ansInts = true;
                break;
            }
        }

        // octree search
        bool octInts = octree.intersects(ray, intersectsFunc);

        EXPECT_EQ(ansInts, octInts);
    }
}

TEST(Octree, ClosestIntersection) {
    Octree<BoundingBox3D> octree;

    auto overlapsFunc = [](const BoundingBox3D& a, const BoundingBox3D& bbox) {
        return bbox.overlaps(a);
    };

    auto intersectsFunc = [](const BoundingBox3D& a, const Ray3D& ray) {
        auto bboxResult = a.closestIntersection(ray);
        if (bboxResult.isIntersecting) {
            return bboxResult.tNear;
        } else {
            return kMaxD;
        }
    };

    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<BoundingBox3D> items(numSamples / 2);
    size_t i = 0;
    std::generate(items.begin(), items.end(), [&]() {
        auto c = getSamplePoints3()[i++];
        BoundingBox3D box(c, c);
        box.expand(0.1);
        return box;
    });

    octree.build(items, BoundingBox3D({0, 0, 0}, {1, 1, 1}), overlapsFunc, 5);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray3D ray(getSamplePoints3()[i + numSamples / 2],
                  getSampleDirs3()[i + numSamples / 2]);
        // ad-hoc search
        ClosestIntersectionQueryResult3<BoundingBox3D> ansInts;
        for (size_t j = 0; j < numSamples / 2; ++j) {
            double dist = intersectsFunc(items[j], ray);
            if (dist < ansInts.distance) {
                ansInts.distance = dist;
                ansInts.item = &octree.item(j);
            }
        }

        // octree search
        auto octInts = octree.closestIntersection(ray, intersectsFunc);

        EXPECT_DOUBLE_EQ(ansInts.distance, octInts.distance);
        EXPECT_EQ(ansInts.item, octInts.item);
    }
}
