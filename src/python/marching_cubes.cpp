// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "marching_cubes.h"
#include "pybind11_utils.h"

#include <jet/marching_cubes.h>
#include <jet/scalar_grid3.h>

namespace py = pybind11;
using namespace jet;

void addMarchingCubes(pybind11::module& m) {
    m.def("marchingCubes",
          [](ScalarGrid3Ptr grid, py::object gridSize, py::object origin,
             double isoValue, int bndFlag) -> TriangleMesh3Ptr {
              auto mesh = TriangleMesh3::builder().makeShared();
              marchingCubes(
                  grid->constDataAccessor(), objectToVector3D(gridSize),
                  objectToVector3D(origin), mesh.get(), isoValue, bndFlag);
              return mesh;
          },
          R"pbdoc(
          Computes marching cubes and extract triangle mesh from grid.
          )pbdoc");
}
