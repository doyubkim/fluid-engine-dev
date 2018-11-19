// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array.h>
#include <jet/bcc_lattice_point_generator.h>
#include <jet/bounding_box.h>
#include <jet/point_hash_grid_searcher.h>
#include <jet/point_hash_grid_utils.h>
#include <jet/point_parallel_hash_grid_searcher.h>
#include <jet/sph_system_data.h>
#include <jet/triangle_point_generator.h>

using namespace jet;

JET_TESTS(PointHashGridSearcher2);

JET_BEGIN_TEST_F(PointHashGridSearcher2, Build) {
    Array1<Vector2D> points;
    TrianglePointGenerator pointsGenerator;
    BoundingBox2D bbox(Vector2D(0, 0), Vector2D(1, 1));
    double spacing = 0.1;

    pointsGenerator.generate(bbox, spacing, &points);

    PointHashGridSearcher2 pointSearcher({4, 4}, 0.18);
    pointSearcher.build(ArrayView1<Vector2D>(points.data(), points.size()));

    Array2<double> grid(4, 4, 0.0);

    for (size_t j = 0; j < grid.size().y; ++j) {
        for (size_t i = 0; i < grid.size().x; ++i) {
            size_t key = PointHashGridUtils2::getHashKeyFromBucketIndex(
                Vector2Z(static_cast<ssize_t>(i), static_cast<ssize_t>(j)),
                {4, 4});
            size_t value = pointSearcher.buckets()[key].length();
            grid(i, j) += static_cast<double>(value);
        }
    }

    saveData(grid.view(), "data_#grid2.npy");
}
JET_END_TEST_F

JET_TESTS(PointHashGridSearcher3);

JET_BEGIN_TEST_F(PointHashGridSearcher3, Build) {
    Array1<Vector3D> points;
    BccLatticePointGenerator pointsGenerator;
    BoundingBox3D bbox(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    double spacing = 0.1;

    pointsGenerator.generate(bbox, spacing, &points);

    PointHashGridSearcher3 pointSearcher({4, 4, 4}, 0.18);
    pointSearcher.build(ArrayView1<Vector3D>(points.data(), points.size()));

    Array2<double> grid(4, 4, 0.0);

    for (size_t j = 0; j < grid.size().y; ++j) {
        for (size_t i = 0; i < grid.size().x; ++i) {
            size_t key = PointHashGridUtils2::getHashKeyFromBucketIndex(
                Vector3Z(static_cast<ssize_t>(i), static_cast<ssize_t>(j), 0),
                {4, 4});
            size_t value = pointSearcher.buckets()[key].length();
            grid(i, j) += static_cast<double>(value);
        }
    }

    saveData(grid.view(), "data_#grid2.npy");
}
JET_END_TEST_F

JET_TESTS(PointParallelHashGridSearcher2);

JET_BEGIN_TEST_F(PointParallelHashGridSearcher2, Build) {
    Array1<Vector2D> points;
    TrianglePointGenerator pointsGenerator;
    BoundingBox2D bbox(Vector2D(0, 0), Vector2D(1, 1));
    double spacing = 0.1;

    pointsGenerator.generate(bbox, spacing, &points);

    PointParallelHashGridSearcher2 pointSearcher({4, 4}, 0.18);
    pointSearcher.build(ArrayView1<Vector2D>(points.data(), points.size()));

    Array2<double> grid(4, 4, 0.0);

    for (size_t j = 0; j < grid.size().y; ++j) {
        for (size_t i = 0; i < grid.size().x; ++i) {
            size_t key = PointHashGridUtils2::getHashKeyFromBucketIndex(
                Vector2Z(static_cast<ssize_t>(i), static_cast<ssize_t>(j)),
                {4, 4});
            size_t start = pointSearcher.startIndexTable()[key];
            size_t end = pointSearcher.endIndexTable()[key];
            size_t value = end - start;
            grid(i, j) += static_cast<double>(value);
        }
    }

    saveData(grid.view(), "data_#grid2.npy");
}
JET_END_TEST_F

JET_TESTS(PointParallelHashGridSearcher3);

JET_BEGIN_TEST_F(PointParallelHashGridSearcher3, Build) {
    Array1<Vector3D> points;
    BccLatticePointGenerator pointsGenerator;
    BoundingBox3D bbox(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    double spacing = 0.1;

    pointsGenerator.generate(bbox, spacing, &points);

    PointParallelHashGridSearcher3 pointSearcher({4, 4, 4}, 0.18);
    pointSearcher.build(ArrayView1<Vector3D>(points.data(), points.size()));

    Array2<double> grid(4, 4, 0.0);

    for (size_t j = 0; j < grid.size().y; ++j) {
        for (size_t i = 0; i < grid.size().x; ++i) {
            size_t key = PointHashGridUtils2::getHashKeyFromBucketIndex(
                Vector3Z(static_cast<ssize_t>(i), static_cast<ssize_t>(j), 0),
                {4, 4});
            size_t start = pointSearcher.startIndexTable()[key];
            size_t end = pointSearcher.endIndexTable()[key];
            size_t value = end - start;
            grid(i, j) += static_cast<double>(value);
        }
    }

    saveData(grid.view(), "data_#grid2.npy");
}
JET_END_TEST_F
