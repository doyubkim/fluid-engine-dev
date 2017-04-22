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
        .def("__init__",
             [](Frame& instance, py::args args, py::kwargs kwargs) {
                 int index = 0;
                 double timeIntervalInSeconds = 1.0 / 60.0;

                 // See if we have list of parameters
                 if (args.size() <= 2) {
                     if (args.size() > 0) {
                         index = args[0].cast<int>();
                     }
                     if (args.size() > 1) {
                         timeIntervalInSeconds = args[1].cast<double>();
                     }
                 } else {
                     throw std::invalid_argument("Too many arguments.");
                 }

                 if (kwargs.contains("index")) {
                     index = kwargs["index"].cast<int>();
                 }
                 if (kwargs.contains("timeIntervalInSeconds")) {
                     timeIntervalInSeconds =
                         kwargs["timeIntervalInSeconds"].cast<double>();
                 }
                 new (&instance) Frame(index, timeIntervalInSeconds);
             },
             "Constructs Frame\n\n"
             "This method constructs Frame with index and "
             "timeIntervalInSeconds.")
        .def_readwrite("index", &Frame::index)
        .def_readwrite("timeIntervalInSeconds", &Frame::timeIntervalInSeconds)
        .def("timeInSeconds", &Frame::timeInSeconds)
        .def("advance",
             [](Frame& instance, int delta) { instance.advance(delta); },
             py::arg("delta") = 1);
}

void addAnimation(pybind11::module& m) {
    py::class_<Animation, AnimationPtr>(m, "Animation")
        .def("update", [](Animation& instance, const Frame& frame) {
            instance.update(frame);
        });
}
