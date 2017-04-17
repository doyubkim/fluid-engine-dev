// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_PYTHON_PYBIND11_UTILS_H_
#define SRC_PYTHON_PYBIND11_UTILS_H_

#include <jet/size2.h>
#include <jet/size3.h>
#include <jet/vector2.h>
#include <jet/vector3.h>
#include <jet/vector4.h>

#include <pybind11/pybind11.h>

namespace jet {

inline Size2 tupleToSize2(pybind11::tuple tpl) {
    Size2 ret;

    if (tpl.size() == 2) {
        for (size_t i = 0; i < 2; ++i) {
            ret[i] = tpl[i].cast<size_t>();
        }
    } else {
        throw std::invalid_argument("Invalid size.");
    }
    return ret;
}

inline Size3 tupleToSize3(pybind11::tuple tpl) {
    Size3 ret;

    if (tpl.size() == 3) {
        for (size_t i = 0; i < 3; ++i) {
            ret[i] = tpl[i].cast<size_t>();
        }
    } else {
        throw std::invalid_argument("Invalid size.");
    }
    return ret;
}

inline pybind11::tuple size2ToTuple(const Size2& sz) {
    return pybind11::make_tuple(sz.x, sz.y);
};

inline pybind11::tuple size3ToTuple(const Size3& sz) {
    return pybind11::make_tuple(sz.x, sz.y, sz.z);
};

template <typename T, size_t N>
inline Vector<T, N> tupleToVector(pybind11::tuple tpl) {
    Vector<T, N> ret;

    if (tpl.size() == N) {
        for (size_t i = 0; i < N; ++i) {
            ret[i] = tpl[i].cast<T>();
        }
    } else {
        throw std::invalid_argument("Invalid size.");
    }
    return ret;
}

inline Vector2D tupleToVector2D(pybind11::tuple tpl) {
    return tupleToVector<double, 2>(tpl);
}

inline Vector3D tupleToVector3D(pybind11::tuple tpl) {
    return tupleToVector<double, 3>(tpl);
}

inline Vector4D tupleToVector4D(pybind11::tuple tpl) {
    return tupleToVector<double, 4>(tpl);
}

template <typename T>
inline pybind11::tuple vector2ToTuple(const Vector<T, 2>& vec) {
    return pybind11::make_tuple(vec.x, vec.y);
};

template <typename T>
inline pybind11::tuple vector3ToTuple(const Vector<T, 3>& vec) {
    return pybind11::make_tuple(vec.x, vec.y, vec.z);
};

template <typename T>
inline pybind11::tuple vector4ToTuple(const Vector<T, 4>& vec) {
    return pybind11::make_tuple(vec.x, vec.y, vec.z, vec.w);
};
}

#endif  // SRC_PYTHON_PYBIND11_UTILS_H_
