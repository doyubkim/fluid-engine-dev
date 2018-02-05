// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "constants.h"
#include "pybind11_utils.h"

#include <jet/constants.h>

namespace py = pybind11;
using namespace jet;

void addConstants(py::module& m) {
    m.attr("DIRECTION_NONE") = py::int_(kDirectionNone);
    m.attr("DIRECTION_LEFT") = py::int_(kDirectionLeft);
    m.attr("DIRECTION_RIGHT") = py::int_(kDirectionRight);
    m.attr("DIRECTION_DOWN") = py::int_(kDirectionDown);
    m.attr("DIRECTION_UP") = py::int_(kDirectionUp);
    m.attr("DIRECTION_BACK") = py::int_(kDirectionBack);
    m.attr("DIRECTION_FRONT") = py::int_(kDirectionFront);
    m.attr("DIRECTION_ALL") = py::int_(kDirectionAll);
}
