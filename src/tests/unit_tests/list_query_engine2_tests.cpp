// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/list_query_engine2.h>

using namespace jet;

TEST(ListQueryEngine2, BoxIntersection) {
    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<Vector2D> points(getSamplePoints2(),
                                 getSamplePoints2() + numSamples);

    ListQueryEngine2<Vector2D> engine;
    engine.add(points);

    auto testFunc = [](const Vector2D& pt, const BoundingBox2D& bbox) {
        return bbox.contains(pt);
    };

    BoundingBox2D testBox({0.25, 0.2}, {0.5, 0.4});
    size_t numIntersections = 0;
    for (size_t i = 0; i < numSamples; ++i) {
        numIntersections += testFunc(getSamplePoints2()[i], testBox);
    }
    bool hasIntersection = numIntersections > 0;

    EXPECT_EQ(hasIntersection, engine.intersects(testBox, testFunc));

    BoundingBox2D testBox2({0.2, 0.2}, {0.6, 0.5});
    numIntersections = 0;
    for (size_t i = 0; i < numSamples; ++i) {
        numIntersections += testFunc(getSamplePoints2()[i], testBox2);
    }
    hasIntersection = numIntersections > 0;

    EXPECT_EQ(hasIntersection, engine.intersects(testBox2, testFunc));

    size_t measured = 0;
    engine.forEachIntersectingItem(testBox2, testFunc, [&](const Vector2D& pt) {
        EXPECT_TRUE(testFunc(pt, testBox2));
        ++measured;
    });

    EXPECT_EQ(numIntersections, measured);
}

TEST(ListQueryEngine2, RayIntersection) {
    ListQueryEngine2<BoundingBox2D> engine;

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

    engine.add(items);

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

        // engine search
        bool engInts = engine.intersects(ray, intersectsFunc);

        EXPECT_EQ(ansInts, engInts);
    }
}

TEST(ListQueryEngine2, ClosestIntersection) {
    ListQueryEngine2<BoundingBox2D> engine;

    auto intersectsFunc = [](const BoundingBox2D& a, const Ray2D& ray) {
        auto bboxResult = a.closestIntersection(ray);
        return bboxResult.tNear;
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

    engine.add(items);

    for (i = 0; i < numSamples / 2; ++i) {
        Ray2D ray(getSamplePoints2()[i + numSamples / 2],
                  getSampleDirs2()[i + numSamples / 2]);
        // ad-hoc search
        ClosestIntersectionQueryResult2<BoundingBox2D> ansInts;
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
            EXPECT_VECTOR2_EQ(ansInts.item->lowerCorner,
                              engInts.item->lowerCorner);
            EXPECT_VECTOR2_EQ(ansInts.item->upperCorner,
                              engInts.item->upperCorner);
        } else {
            EXPECT_EQ(nullptr, ansInts.item);
            EXPECT_EQ(nullptr, engInts.item);
        }
        EXPECT_DOUBLE_EQ(ansInts.distance, engInts.distance);
    }
}

TEST(ListQueryEngine2, NearestNeighbor) {
    ListQueryEngine2<Vector2D> engine;

    auto distanceFunc = [](const Vector2D& a, const Vector2D& b) {
        return a.distanceTo(b);
    };

    size_t numSamples = getNumberOfSamplePoints2();
    std::vector<Vector2D> points(getSamplePoints2(),
                                 getSamplePoints2() + numSamples);

    engine.add(points);

    Vector2D testPt(0.5, 0.5);
    auto closest = engine.nearest(testPt, distanceFunc);
    Vector2D answer = getSamplePoints2()[0];
    double bestDist = testPt.distanceTo(answer);
    for (size_t i = 1; i < numSamples; ++i) {
        double dist = testPt.distanceTo(getSamplePoints2()[i]);
        if (dist < bestDist) {
            bestDist = dist;
            answer = getSamplePoints2()[i];
        }
    }

    EXPECT_VECTOR2_EQ(answer, (*(closest.item)));
}
