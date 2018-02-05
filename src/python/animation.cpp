// Copyright (c) 2018 Doyub Kim
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
    py::class_<Frame>(m, "Frame", R"pbdoc(
        Representation of an animation frame.

        This struct holds current animation frame index and frame interval in
        seconds.
        )pbdoc")
        .def("__init__",
             [](Frame& instance, int index, double timeIntervalInSeconds) {
                 new (&instance) Frame(index, timeIntervalInSeconds);
             },
             R"pbdoc(
             Constructs Frame

             This method constructs Frame with index and time interval (in seconds).

             Parameters
             ----------
             - index : The index (default is 0).
             - timeIntervalInSeconds : The time interval in seconds (default is 1.0 / 60.0).
             )pbdoc",
             py::arg("index") = 0,
             py::arg("timeIntervalInSeconds") = 1.0 / 60.0)
        .def_readwrite("index", &Frame::index,
                       R"pbdoc(Index of the frame)pbdoc")
        .def_readwrite("timeIntervalInSeconds", &Frame::timeIntervalInSeconds,
                       R"pbdoc(Time interval of the frame in seconds)pbdoc")
        .def("timeInSeconds", &Frame::timeInSeconds,
             R"pbdoc(Elapsed time in seconds)pbdoc")
        .def("advance",
             [](Frame& instance, int delta) { instance.advance(delta); },
             R"pbdoc(
             Advances multiple frames.

             Parameters
             ----------
             - delta : Number of frames to advance.
             )pbdoc",
             py::arg("delta") = 1);
}

class PyAnimation : public Animation {
 public:
    using Animation::Animation;

    void onUpdate(const Frame& frame) override {
        PYBIND11_OVERLOAD_PURE(void, Animation, onUpdate, frame);
    }
};

void addAnimation(pybind11::module& m) {
    py::class_<Animation, PyAnimation, AnimationPtr>(m, "Animation", R"pbdoc(
        Abstract base class for animation-related class.

        This class represents the animation logic in very abstract level.
        Generally animation is a function of time and/or its previous state.
        This base class provides a virtual function update() which can be
        overriden by its sub-classes to implement their own state update logic.
        )pbdoc")
        .def(py::init<>())
        .def("update", &Animation::update,
             R"pbdoc(
             Updates animation state for given `frame`.

             Parameters
             ----------
             - frame : Number of frames to advance.
             )pbdoc",
             py::arg("frame"));
}
