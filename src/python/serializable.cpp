// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "serializable.h"
#include "pybind11_utils.h"

#include <jet/serialization.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace jet;

void addSerializable(py::module& m) {
    py::class_<Serializable, std::shared_ptr<Serializable>>(m, "Serializable",
                                                            R"pbdoc(
        Abstract base class for any serializable class.
        )pbdoc")
        .def("serialize",
             [](const Serializable& instance) {
                 std::vector<uint8_t> buffer;
                 instance.serialize(&buffer);
                 return buffer;
             },
             R"pbdoc(
             Serializes this instance into the flat buffer.
             )pbdoc")
        .def("serializeToFile",
             [](const Serializable& instance, const std::string& filename) {
                 std::vector<uint8_t> buffer;
                 instance.serialize(&buffer);
                 std::ofstream file(filename.c_str(), std::ios::binary);
                 if (file) {
                     file.write(reinterpret_cast<char*>(buffer.data()),
                                buffer.size());
                     file.close();
                 } else {
                     throw std::invalid_argument("Cannot write to file.");
                 }
             },
             R"pbdoc(
             Serializes this instance into the file.
             )pbdoc",
             py::arg("filename"))
        .def("deserialize", &Serializable::deserialize,
             R"pbdoc(
             Deserializes this instance from the flat buffer.
             )pbdoc",
             py::arg("buffer"))
        .def("deserializeFromFile", [](Serializable& instance, const std::string& filename) {
                 std::ifstream file(filename.c_str(), std::ios::binary);
                 if (file) {
                     std::vector<uint8_t> buffer(
                             (std::istreambuf_iterator<char>(file)),
                             (std::istreambuf_iterator<char>()));
                     instance.deserialize(buffer);
                     file.close();
                 } else {
                     throw std::invalid_argument("Cannot write to file.");
                 }
             },
             R"pbdoc(
             Deserializes this instance from the flat buffer.
             )pbdoc",
             py::arg("filename"));
}
