// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array3.h>
#include <jet/array_utils.h>
#include <jet/triangle_mesh_to_sdf.h>

#include <algorithm>
#include <vector>

using namespace jet;

namespace jet {

void triangleMeshToSdf(const TriangleMesh3& mesh, ScalarGrid3* sdf,
                       const unsigned int) {
    Size3 size = sdf->dataSize();
    if (size.x * size.y * size.z == 0) {
        return;
    }

    const auto pos = sdf->dataPosition();
    mesh.updateQueryEngine();
    sdf->parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        const Vector3D p = pos(i, j, k);
        const double d = mesh.closestDistance(p);
        const double sd = mesh.isInside(p) ? -d : d;
        (*sdf)(i, j, k) = sd;
    });
}

}  // namespace jet
