// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "animation.h"
#include "pybind11_utils.h"

#include <jet/animation.h>

namespace py = pybind11;
using namespace jet;

void addFrame(pybind11::module& m) {
    py::class_<Frame>(m, "Frame")
        .def(py::init<int32_t, double>())
        .def_readwrite("index", &Frame::index)
        .def_readwrite("timeIntervalInSeconds", &Frame::timeIntervalInSeconds)
        .def("timeInSeconds", &Frame::timeInSeconds)
        .def("advance",
             [](Frame& instance, int32_t delta) { instance.advance(delta); },
             py::arg("delta") = 1);
}

void addAnimation(pybind11::module& m) {
    (void)m;
}
