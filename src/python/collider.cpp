// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "collider.h"
#include "pybind11_utils.h"

#include <jet/collider3.h>

namespace py = pybind11;
using namespace jet;

void addCollider3(py::module& m) {
    py::class_<Collider3, Collider3Ptr>(m, "Collider3");
}
