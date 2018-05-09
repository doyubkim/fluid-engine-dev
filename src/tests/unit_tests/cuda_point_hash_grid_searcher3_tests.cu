// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <gtest/gtest.h>

#include <jet/array1.h>
#include <jet/cuda_array1.h>
#include <jet/cuda_point_hash_grid_searcher3.h>
#include <jet/point_parallel_hash_grid_searcher3.h>

#include <cuda_runtime.h>

#include <vector>

using namespace jet;

namespace {

struct ForEachCallback {
    float4* points;
    int* isValid;
    int* visited;

    JET_CUDA_HOST_DEVICE void operator()(size_t i, float4 o, size_t j,
                                         float4 pt) {
        visited[j] = 1;

        if (j == 1) {
            isValid[j] = 0;
            return;
        }

        if (j == 0) {
            isValid[j] = points[0] == pt;
        } else if (j == 2) {
            isValid[j] = points[2] == pt;
        }
    }
};

}  // namespace

TEST(CudaPointHashGridSearcher3, Build) {
    // CPU baseline
    Array1<Vector3D> points = {Vector3D(0, 1, 3), Vector3D(2, 5, 4),
                               Vector3D(-1, 3, 0)};

    PointParallelHashGridSearcher3 searcher(4, 4, 4, std::sqrt(10));
    searcher.build(points.accessor());

    // GPU
    CudaArray1<float4> pointsD(3);
    pointsD[0] = make_float4(0, 1, 3, 0);
    pointsD[1] = make_float4(2, 5, 4, 0);
    pointsD[2] = make_float4(-1, 3, 0, 0);

    CudaPointHashGridSearcher3 searcherD(4, 4, 4, std::sqrt(10.0f));
    searcherD.build(pointsD.view());

    // Compare
    EXPECT_EQ(searcher.keys().size(), searcherD.keys().size());
    EXPECT_EQ(searcher.startIndexTable().size(),
              searcherD.startIndexTable().size());
    EXPECT_EQ(searcher.endIndexTable().size(),
              searcherD.endIndexTable().size());
    EXPECT_EQ(searcher.sortedIndices().size(),
              searcherD.sortedIndices().size());

    for (size_t i = 0; i < searcher.keys().size(); ++i) {
        uint32_t valD = searcherD.keys()[i];
        EXPECT_EQ(searcher.keys()[i], valD);
    }

    for (size_t i = 0; i < searcher.startIndexTable().size(); ++i) {
        uint32_t valD = searcherD.startIndexTable()[i];
        if (valD == 0xffffffff) {
            EXPECT_EQ(kMaxSize, searcher.startIndexTable()[i]);
        } else {
            EXPECT_EQ(searcher.startIndexTable()[i], valD);
        }
    }

    for (size_t i = 0; i < searcher.endIndexTable().size(); ++i) {
        uint32_t valD = searcherD.endIndexTable()[i];
        if (valD == 0xffffffff) {
            EXPECT_EQ(kMaxSize, searcher.endIndexTable()[i]);
        } else {
            EXPECT_EQ(searcher.endIndexTable()[i], valD);
        }
    }

    for (size_t i = 0; i < searcher.sortedIndices().size(); ++i) {
        size_t valD = searcherD.sortedIndices()[i];
        EXPECT_EQ(searcher.sortedIndices()[i], valD);
    }
}

TEST(CudaPointHashGridSearcher3, ForEachNearbyPoint) {
    CudaArray1<float4> pointsD(3);
    pointsD[0] = make_float4(0, 1, 3, 0);
    pointsD[1] = make_float4(2, 5, 4, 0);
    pointsD[2] = make_float4(-1, 2.9f, 0, 0);

    CudaArray1<float4> origins(1, make_float4(0, 0, 0, 0));
    CudaArray1<int> isValid(3, 1);
    CudaArray1<int> visited(3, 0);

    CudaPointHashGridSearcher3 searcherD(4, 4, 4, std::sqrt(10.0f));
    searcherD.build(pointsD.view());

    ForEachCallback func;
    func.points = pointsD.data();
    func.isValid = isValid.data();
    func.visited = visited.data();

    searcherD.forEachNearbyPoint(origins.view(), std::sqrt(10.0f), func);

    int iv = isValid[0];
    int vd = visited[0];
    EXPECT_EQ(1, iv);
    EXPECT_EQ(1, vd);
    iv = isValid[1];
    vd = visited[1];
    EXPECT_EQ(1, iv);
    EXPECT_EQ(0, vd);
    iv = isValid[2];
    vd = visited[2];
    EXPECT_EQ(1, iv);
    EXPECT_EQ(1, vd);
}
