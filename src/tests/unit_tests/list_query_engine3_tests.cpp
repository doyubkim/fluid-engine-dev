// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/list_query_engine3.h>

using namespace jet;

TEST(ListQueryEngine3, BoxIntersection) {
    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<Vector3D> points(getSamplePoints3(), getSamplePoints3() + numSamples);

    ListQueryEngine3<Vector3D> engine;
    engine.add(points);

    auto testFunc = [](const Vector3D& pt, const BoundingBox3D& bbox) {
        return bbox.contains(pt);
    };

    BoundingBox3D testBox({0.25, 0.15, 0.3}, {0.5, 0.6, 0.4});
    size_t numIntersections = 0;
    for (size_t i = 0; i < numSamples; ++i) {
        numIntersections += testFunc(getSamplePoints3()[i], testBox);
    }
    bool hasIntersection = numIntersections > 0;

    EXPECT_EQ(hasIntersection, engine.intersects(testBox, testFunc));

    BoundingBox3D testBox2({0.3, 0.2, 0.1}, {0.6, 0.5, 0.4});
    numIntersections = 0;
    for (size_t i = 0; i < numSamples; ++i) {
        numIntersections += testFunc(getSamplePoints3()[i], testBox2);
    }
    hasIntersection = numIntersections > 0;

    EXPECT_EQ(hasIntersection, engine.intersects(testBox2, testFunc));

    size_t measured = 0;
    engine.forEachIntersectingItem(testBox2, testFunc, [&](const Vector3D& pt) {
        EXPECT_TRUE(testFunc(pt, testBox2));
        ++measured;
    });

    EXPECT_EQ(numIntersections, measured);
}

TEST(ListQueryEngine3, RayIntersection) {
    ListQueryEngine3<BoundingBox3D> engine;

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

    engine.add(items);

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

        // engine search
        bool engInts = engine.intersects(ray, intersectsFunc);

        EXPECT_EQ(ansInts, engInts);
    }
}

TEST(ListQueryEngine3, ClosestIntersection) {
    ListQueryEngine3<BoundingBox3D> engine;

    auto intersectsFunc = [](const BoundingBox3D& a, const Ray3D& ray) {
        auto bboxResult = a.closestIntersection(ray);
        return bboxResult.tNear;
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

    engine.add(items);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray3D ray(getSamplePoints3()[i + numSamples / 2],
                  getSampleDirs3()[i + numSamples / 2]);
        // ad-hoc search
        ClosestIntersectionQueryResult3<BoundingBox3D> ansInts;
        for (size_t j = 0; j < numSamples / 2; ++j) {
            double dist = intersectsFunc(items[j], ray);
            if (dist < ansInts.distance) {
                ansInts.distance = dist;
                ansInts.item = &items[j];
            }
        }

        // engine search
        auto engInts = engine.closestIntersection(ray, intersectsFunc);

        if (ansInts.item != nullptr && engInts.item != nullptr) {
            EXPECT_VECTOR3_EQ(ansInts.item->lowerCorner,
                              engInts.item->lowerCorner);
            EXPECT_VECTOR3_EQ(ansInts.item->upperCorner,
                              engInts.item->upperCorner);
        } else {
            EXPECT_EQ(nullptr, ansInts.item);
            EXPECT_EQ(nullptr, engInts.item);
        }
        EXPECT_DOUBLE_EQ(ansInts.distance, engInts.distance);
    }
}

TEST(ListQueryEngine3, NearestNeighbor) {
    ListQueryEngine3<Vector3D> engine;

    auto distanceFunc = [](const Vector3D& a, const Vector3D& b) {
        return a.distanceTo(b);
    };

    size_t numSamples = getNumberOfSamplePoints3();
    std::vector<Vector3D> points(getSamplePoints3(), getSamplePoints3() + numSamples);

    engine.add(points);

    Vector3D testPt(0.5, 0.5, 0.5);
    auto closest = engine.nearest(testPt, distanceFunc);
    Vector3D answer = getSamplePoints3()[0];
    double bestDist = testPt.distanceTo(answer);
    for (size_t i = 1; i < numSamples; ++i) {
        double dist = testPt.distanceTo(getSamplePoints3()[i]);
        if (dist < bestDist) {
            bestDist = dist;
            answer = getSamplePoints3()[i];
        }
    }

    EXPECT_VECTOR3_EQ(answer, (*(closest.item)));
}
