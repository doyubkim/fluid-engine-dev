// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "grid_emitter.h"
#include "pybind11_utils.h"

#include <jet/grid_emitter2.h>
#include <jet/grid_emitter3.h>

namespace py = pybind11;
using namespace jet;

void addGridEmitter2(py::module& m) {
    py::class_<GridEmitter2, GridEmitter2Ptr>(
        m, "GridEmitter2",
        R"pbdoc(Abstract base class for 2-D grid-based emitters.)pbdoc")
        .def("update", &GridEmitter2::update,
             R"pbdoc(
            Updates the emitter state from `currentTimeInSeconds` to by
            `timeIntervalInSeconds`.

            Parameters
            ----------
            - currentTimeInSeconds : Starting time stamp.
            - timeIntervalInSeconds : Time-step to advance.
            )pbdoc",
             py::arg("currentTimeInSeconds"), py::arg("timeIntervalInSeconds"))
        .def_property(
            "isEnabled", &GridEmitter2::isEnabled, &GridEmitter2::setIsEnabled,
            R"pbdoc(True/false if the emitter is enabled/disabled.)pbdoc")
        .def("setOnBeginUpdateCallback",
             [](GridEmitter2& instance, py::function callback) {
                 instance.setOnBeginUpdateCallback(callback);
             },
             R"pbdoc(
             Sets the callback function to be called when `update` is invoked.

             The callback function takes current simulation time in seconds unit. Use
             this callback to track any motion or state changes related to this
             emitter.

             Parameters
             ----------
             - callback : The callback function.
             )pbdoc",
             py::arg("callback"));
}

void addGridEmitter3(py::module& m) {
    py::class_<GridEmitter3, GridEmitter3Ptr>(
        m, "GridEmitter3",
        R"pbdoc(Abstract base class for 3-D grid-based emitters.)pbdoc")
        .def("update", &GridEmitter3::update,
             R"pbdoc(
            Updates the emitter state from `currentTimeInSeconds` to by
            `timeIntervalInSeconds`.

            Parameters
            ----------
            - currentTimeInSeconds : Starting time stamp.
            - timeIntervalInSeconds : Time-step to advance.
            )pbdoc",
             py::arg("currentTimeInSeconds"), py::arg("timeIntervalInSeconds"))
        .def_property(
            "isEnabled", &GridEmitter3::isEnabled, &GridEmitter3::setIsEnabled,
            R"pbdoc(True/false if the emitter is enabled/disabled.)pbdoc")
        .def("setOnBeginUpdateCallback",
             [](GridEmitter3& instance, py::function callback) {
                 instance.setOnBeginUpdateCallback(callback);
             },
             R"pbdoc(
             Sets the callback function to be called when `update` is invoked.

             The callback function takes current simulation time in seconds unit. Use
             this callback to track any motion or state changes related to this
             emitter.

             Parameters
             ----------
             - callback : The callback function.
             )pbdoc",
             py::arg("callback"));
}
