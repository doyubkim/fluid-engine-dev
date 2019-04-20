// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/bvh3.h>

using namespace jet;

TEST(Bvh3, Constructors) {
    Bvh3<Vector3D> bvh;
    EXPECT_EQ(bvh.begin(), bvh.end());
}

TEST(Bvh3, BasicGetters) {
    Bvh3<Vector3D> bvh;

    std::vector<Vector3D> points{Vector3D(0, 0, 0), Vector3D(1, 1, 1)};
    std::vector<BoundingBox3D> bounds(points.size());
    size_t i = 0;
    BoundingBox3D rootBounds;
    std::generate(bounds.begin(), bounds.end(), [&]() {
        auto c = points[i++];
        BoundingBox3D box(c, c);
        box.expand(0.1);
        rootBounds.merge(box);
        return box;
    });

    bvh.build(points, bounds);

    EXPECT_EQ(2u, bvh.numberOfItems());
    EXPECT_VECTOR3_EQ(points[0], bvh.item(0));
    EXPECT_VECTOR3_EQ(points[1], bvh.item(1));
    EXPECT_EQ(3u, bvh.numberOfNodes());
    EXPECT_EQ(1u, bvh.children(0).first);
    EXPECT_EQ(2u, bvh.children(0).second);
    EXPECT_FALSE(bvh.isLeaf(0));
    EXPECT_TRUE(bvh.isLeaf(1));
    EXPECT_TRUE(bvh.isLeaf(2));
    EXPECT_BOUNDING_BOX3_EQ(rootBounds, bvh.nodeBound(0));
    EXPECT_BOUNDING_BOX3_EQ(bounds[0], bvh.nodeBound(1));
    EXPECT_BOUNDING_BOX3_EQ(bounds[1], bvh.nodeBound(2));
    EXPECT_EQ(bvh.end(), bvh.itemOfNode(0));
    EXPECT_EQ(bvh.begin(), bvh.itemOfNode(1));
    EXPECT_EQ(bvh.begin() + 1, bvh.itemOfNode(2));
}

TEST(Bvh3, Nearest) {
    Bvh3<Vector3D> bvh;

    auto distanceFunc = [](const Vector3D& a, const Vector3D& b) {
        return a.distanceTo(b);
    };

    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<Vector3D> points(getSamplePoints3(), getSamplePoints3() + numSamples);

    std::vector<BoundingBox3D> bounds(points.size());
    size_t i = 0;
    std::generate(bounds.begin(), bounds.end(), [&]() {
        auto c = points[i++];
        BoundingBox3D box(c, c);
        box.expand(0.1);
        return box;
    });

    bvh.build(points, bounds);

    Vector3D testPt(0.5, 0.5, 0.5);
    auto nearest = bvh.nearest(testPt, distanceFunc);
    ptrdiff_t answerIdx = 0;
    double bestDist = testPt.distanceTo(points[answerIdx]);
    for (i = 1; i < numSamples; ++i) {
        double dist = testPt.distanceTo(getSamplePoints3()[i]);
        if (dist < bestDist) {
            bestDist = dist;
            answerIdx = i;
        }
    }

    EXPECT_EQ(answerIdx, nearest.item - &bvh.item(0));
}

TEST(Bvh3, BBoxIntersects) {
    Bvh3<Vector3D> bvh;

    auto overlapsFunc = [](const Vector3D& pt, const BoundingBox3D& bbox) {
        BoundingBox3D box(pt, pt);
        box.expand(0.1);
        return bbox.overlaps(box);
    };

    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<Vector3D> points(getSamplePoints3(), getSamplePoints3() + numSamples);

    std::vector<BoundingBox3D> bounds(points.size());
    size_t i = 0;
    std::generate(bounds.begin(), bounds.end(), [&]() {
        auto c = points[i++];
        BoundingBox3D box(c, c);
        box.expand(0.1);
        return box;
    });

    bvh.build(points, bounds);

    BoundingBox3D testBox({0.25, 0.15, 0.3}, {0.5, 0.6, 0.4});
    bool hasOverlaps = false;
    for (i = 0; i < numSamples; ++i) {
        hasOverlaps |= overlapsFunc(getSamplePoints3()[i], testBox);
    }

    EXPECT_EQ(hasOverlaps, bvh.intersects(testBox, overlapsFunc));

    BoundingBox3D testBox2({0.3, 0.2, 0.1}, {0.6, 0.5, 0.4});
    hasOverlaps = false;
    for (i = 0; i < numSamples; ++i) {
        hasOverlaps |= overlapsFunc(getSamplePoints3()[i], testBox2);
    }

    EXPECT_EQ(hasOverlaps, bvh.intersects(testBox2, overlapsFunc));
}

TEST(Bvh3, RayIntersects) {
    Bvh3<BoundingBox3D> bvh;

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

    bvh.build(items, items);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray3D ray(getSampleDirs3()[i + numSamples / 2],
                  getSampleDirs3()[i + numSamples / 2]);
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

TEST(Bvh3, ClosestIntersection) {
    Bvh3<BoundingBox3D> bvh;

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

    bvh.build(items, items);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray3D ray(getSamplePoints3()[i + numSamples / 2],
                  getSampleDirs3()[i + numSamples / 2]);
        // ad-hoc search
        ClosestIntersectionQueryResult3<BoundingBox3D> ansInts;
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

TEST(Bvh3, ForEachOverlappingItems) {
    Bvh3<Vector3D> bvh;

    auto overlapsFunc = [](const Vector3D& pt, const BoundingBox3D& bbox) {
        return bbox.contains(pt);
    };

    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<Vector3D> points(getSamplePoints3(), getSamplePoints3() + numSamples);

    std::vector<BoundingBox3D> bounds(points.size());
    size_t i = 0;
    std::generate(bounds.begin(), bounds.end(), [&]() {
        auto c = points[i++];
        BoundingBox3D box(c, c);
        box.expand(0.1);
        return box;
    });

    bvh.build(points, bounds);

    BoundingBox3D testBox({0.3, 0.2, 0.1}, {0.6, 0.5, 0.4});
    size_t numOverlaps = 0;
    for (i = 0; i < numSamples; ++i) {
        numOverlaps += overlapsFunc(getSamplePoints3()[i], testBox);
    }

    size_t measured = 0;
    bvh.forEachIntersectingItem(testBox, overlapsFunc, [&](const Vector3D& pt) {
        EXPECT_TRUE(overlapsFunc(pt, testBox));
        ++measured;
    });

    EXPECT_EQ(numOverlaps, measured);
}
