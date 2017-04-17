// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "animation.h"
#include "flip_solver3.h"
#include "logging.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_PLUGIN(pyjet) {
    py::module m("pyjet",
                 "Fluid simulation engine for computer graphics applications");

    addAnimation(m);
    addFlipSolver3(m);
    addFrame(m);
    addLogging(m);

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}