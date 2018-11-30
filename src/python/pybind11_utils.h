// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_PYTHON_PYBIND11_UTILS_H_
#define SRC_PYTHON_PYBIND11_UTILS_H_

#include <jet/matrix.h>
#include <jet/quaternion.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace jet {

inline Vector2UZ tupleToVector2UZ(pybind11::tuple tpl) {
    Vector2UZ ret;

    if (tpl.size() == 2) {
        for (size_t i = 0; i < 2; ++i) {
            ret[i] = tpl[i].cast<size_t>();
        }
    } else {
        throw std::invalid_argument("Invalid size.");
    }
    return ret;
}

inline Vector2UZ tupleToVector2UZ(pybind11::list lst) {
    Vector2UZ ret;

    if (lst.size() == 2) {
        for (size_t i = 0; i < 2; ++i) {
            ret[i] = lst[i].cast<size_t>();
        }
    } else {
        throw std::invalid_argument("Invalid size.");
    }
    return ret;
}

inline Vector3UZ tupleToVector3UZ(pybind11::tuple tpl) {
    Vector3UZ ret;

    if (tpl.size() == 3) {
        for (size_t i = 0; i < 3; ++i) {
            ret[i] = tpl[i].cast<size_t>();
        }
    } else {
        throw std::invalid_argument("Invalid size.");
    }
    return ret;
}

inline Vector3UZ tupleToVector3UZ(pybind11::list lst) {
    Vector3UZ ret;

    if (lst.size() == 3) {
        for (size_t i = 0; i < 3; ++i) {
            ret[i] = lst[i].cast<size_t>();
        }
    } else {
        throw std::invalid_argument("Invalid size.");
    }
    return ret;
}

inline pybind11::tuple size2ToTuple(const Vector2UZ& sz) {
    return pybind11::make_tuple(sz.x, sz.y);
};

inline pybind11::tuple size3ToTuple(const Vector3UZ& sz) {
    return pybind11::make_tuple(sz.x, sz.y, sz.z);
};

////////////////////////////////////////////////////////////////////////////////

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

template <typename T, size_t N>
inline Vector<T, N> tupleToVector(pybind11::list tpl) {
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

template <typename T>
inline Quaternion<T> tupleToQuaternion(pybind11::tuple tpl) {
    Quaternion<T> ret;

    for (size_t i = 0; i < 4; ++i) {
        ret[i] = tpl[i].cast<T>();
    }

    return ret;
}

template <typename T>
inline Quaternion<T> tupleToQuaternion(pybind11::list tpl) {
    Quaternion<T> ret;

    for (size_t i = 0; i < 4; ++i) {
        ret[i] = tpl[i].cast<T>();
    }

    return ret;
}

inline Vector2F tupleToVector2F(pybind11::tuple tpl) {
    return tupleToVector<float, 2>(tpl);
}

inline Vector2F tupleToVector2F(pybind11::list tpl) {
    return tupleToVector<float, 2>(tpl);
}

inline Vector3F tupleToVector3F(pybind11::tuple tpl) {
    return tupleToVector<float, 3>(tpl);
}

inline Vector3F tupleToVector3F(pybind11::list tpl) {
    return tupleToVector<float, 3>(tpl);
}

inline Vector4F tupleToVector4F(pybind11::tuple tpl) {
    return tupleToVector<float, 4>(tpl);
}

inline Vector4F tupleToVector4F(pybind11::list tpl) {
    return tupleToVector<float, 4>(tpl);
}

inline QuaternionF tupleToQuaternionF(pybind11::tuple tpl) {
    return tupleToQuaternion<float>(tpl);
}

inline QuaternionF tupleToQuaternionF(pybind11::list tpl) {
    return tupleToQuaternion<float>(tpl);
}

inline Vector2D tupleToVector2D(pybind11::tuple tpl) {
    return tupleToVector<double, 2>(tpl);
}

inline Vector2D tupleToVector2D(pybind11::list tpl) {
    return tupleToVector<double, 2>(tpl);
}

inline Vector3D tupleToVector3D(pybind11::tuple tpl) {
    return tupleToVector<double, 3>(tpl);
}

inline Vector3D tupleToVector3D(pybind11::list tpl) {
    return tupleToVector<double, 3>(tpl);
}

inline Vector4D tupleToVector4D(pybind11::tuple tpl) {
    return tupleToVector<double, 4>(tpl);
}

inline Vector4D tupleToVector4D(pybind11::list tpl) {
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

////////////////////////////////////////////////////////////////////////////////

inline QuaternionD tupleToQuaternionD(pybind11::tuple tpl) {
    return tupleToQuaternion<double>(tpl);
}

inline QuaternionD tupleToQuaternionD(pybind11::list tpl) {
    return tupleToQuaternion<double>(tpl);
}

////////////////////////////////////////////////////////////////////////////////

inline Vector2UZ objectToVector2UZ(const pybind11::object& obj) {
    if (pybind11::isinstance<Vector2UZ>(obj)) {
        return obj.cast<Vector2UZ>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector2UZ(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector2UZ(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to Vector2UZ.");
    }
}

inline Vector3UZ objectToVector3UZ(const pybind11::object& obj) {
    if (pybind11::isinstance<Vector3UZ>(obj)) {
        return obj.cast<Vector3UZ>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector3UZ(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector3UZ(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to Vector3UZ.");
    }
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

inline Vector2F objectToVector2F(const pybind11::object& obj) {
    if (pybind11::isinstance<Vector2F>(obj)) {
        return obj.cast<Vector2F>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector2F(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector2F(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to Vector2F.");
    }
}

inline Vector2D objectToVector2D(const pybind11::object& obj) {
    if (pybind11::isinstance<Vector2D>(obj)) {
        return obj.cast<Vector2D>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector2D(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector2D(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to Vector2D.");
    }
}

inline Vector3F objectToVector3F(const pybind11::object& obj) {
    if (pybind11::isinstance<Vector3F>(obj)) {
        return obj.cast<Vector3F>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector3F(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector3F(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to Vector3F.");
    }
}

inline Vector3D objectToVector3D(const pybind11::object& obj) {
    if (pybind11::isinstance<Vector3D>(obj)) {
        return obj.cast<Vector3D>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector3D(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector3D(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to Vector3D.");
    }
}

inline Vector4F objectToVector4F(const pybind11::object& obj) {
    if (pybind11::isinstance<Vector4F>(obj)) {
        return obj.cast<Vector4F>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector4F(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector4F(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to Vector4F.");
    }
}

inline Vector4D objectToVector4D(const pybind11::object& obj) {
    if (pybind11::isinstance<Vector4D>(obj)) {
        return obj.cast<Vector4D>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector4D(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector4D(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to Vector4D.");
    }
}

inline QuaternionF objectToQuaternionF(const pybind11::object& obj) {
    if (pybind11::isinstance<QuaternionF>(obj)) {
        return obj.cast<QuaternionF>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToQuaternionF(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToQuaternionF(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to QuaternionF.");
    }
}

inline QuaternionD objectToQuaternionD(const pybind11::object& obj) {
    if (pybind11::isinstance<QuaternionD>(obj)) {
        return obj.cast<QuaternionD>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToQuaternionD(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToQuaternionD(pybind11::list(obj));
    } else {
        throw std::invalid_argument("Cannot convert to QuaternionD.");
    }
}

////////////////////////////////////////////////////////////////////////////////

inline void parseGridResizeParams(pybind11::args args, pybind11::kwargs kwargs,
                                  Vector2UZ& resolution, Vector2D& gridSpacing,
                                  Vector2D& gridOrigin) {
    // See if we have list of parameters
    if (args.size() <= 3) {
        if (args.size() > 0) {
            resolution = objectToVector2UZ(pybind11::object(args[0]));
        }
        if (args.size() > 1) {
            gridSpacing = objectToVector2D(pybind11::object(args[1]));
        }
        if (args.size() > 2) {
            gridOrigin = objectToVector2D(pybind11::object(args[2]));
        }
    } else {
        throw std::invalid_argument("Too many arguments.");
    }

    // Parse out keyword args
    if (kwargs.contains("resolution")) {
        resolution = objectToVector2UZ(pybind11::object(kwargs["resolution"]));
    }
    if (kwargs.contains("gridSpacing")) {
        gridSpacing = objectToVector2D(pybind11::object(kwargs["gridSpacing"]));
    }
    if (kwargs.contains("gridOrigin")) {
        gridOrigin = objectToVector2D(pybind11::object(kwargs["gridOrigin"]));
    }
    if (kwargs.contains("domainSizeX")) {
        double domainSizeX = kwargs["domainSizeX"].cast<double>();
        gridSpacing.fill(domainSizeX / static_cast<double>(resolution.x));
    }
}

inline void parseGridResizeParams(pybind11::args args, pybind11::kwargs kwargs,
                                  Vector3UZ& resolution, Vector3D& gridSpacing,
                                  Vector3D& gridOrigin) {
    // See if we have list of parameters
    if (args.size() <= 3) {
        if (args.size() > 0) {
            resolution = objectToVector3UZ(pybind11::object(args[0]));
        }
        if (args.size() > 1) {
            gridSpacing = objectToVector3D(pybind11::object(args[1]));
        }
        if (args.size() > 2) {
            gridOrigin = objectToVector3D(pybind11::object(args[2]));
        }
    } else {
        throw std::invalid_argument("Too many arguments.");
    }

    // Parse out keyword args
    if (kwargs.contains("resolution")) {
        resolution = objectToVector3UZ(pybind11::object(kwargs["resolution"]));
    }
    if (kwargs.contains("gridSpacing")) {
        gridSpacing = objectToVector3D(pybind11::object(kwargs["gridSpacing"]));
    }
    if (kwargs.contains("gridOrigin")) {
        gridOrigin = objectToVector3D(pybind11::object(kwargs["gridOrigin"]));
    }
    if (kwargs.contains("domainSizeX")) {
        double domainSizeX = kwargs["domainSizeX"].cast<double>();
        gridSpacing.fill(domainSizeX / static_cast<double>(resolution.x));
    }
}

}  // namespace jet

#define JET_PYTHON_MAKE_INDEX_FUNCTION2(Class, Func)                        \
    [](const Class& instance, py::args args) {                              \
        if (args.size() == 1) {                                             \
            return instance.Func(                                           \
                objectToVector2UZ(args[0]));                                \
        } else {                                                            \
            throw std::invalid_argument("Invalid number of arguments.");    \
        }                                                                   \
    }

#define JET_PYTHON_MAKE_INDEX_FUNCTION3(Class, Func)                        \
    [](const Class& instance, py::args args) {                              \
        if (args.size() == 1) {                                             \
            return instance.Func(                                           \
                objectToVector3UZ(args[0]));                                \
        } else {                                                            \
            throw std::invalid_argument("Invalid number of arguments.");    \
        }                                                                   \
    }

#endif  // SRC_PYTHON_PYBIND11_UTILS_H_
