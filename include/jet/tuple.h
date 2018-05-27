// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TUPLE_H_
#define INCLUDE_JET_TUPLE_H_

#include <jet/macros.h>

#include <array>
#include <cstdint>

namespace jet {

template <typename T, size_t N>
class Tuple {
 public:
    static_assert(N > 0,
                  "Size of static-sized Tuple should be greater than zero.");

    constexpr Tuple() : _elements{} {}

    template <typename... Args>
    constexpr Tuple(const Args&... args) : _elements{{args...}} {}

    T& operator[](size_t i);

    const T& operator[](size_t i) const;

 private:
    std::array<T, N> _elements;
};

template <size_t N>
using ByteN = Tuple<int8_t, N>;

template <size_t N>
using UByteN = Tuple<uint8_t, N>;

template <size_t N>
using ShortN = Tuple<int16_t, N>;

template <size_t N>
using UShortN = Tuple<uint16_t, N>;

template <size_t N>
using IntN = Tuple<int32_t, N>;

template <size_t N>
using UIntN = Tuple<uint32_t, N>;

template <size_t N>
using LongN = Tuple<int64_t, N>;

template <size_t N>
using ULongN = Tuple<uint64_t, N>;

template <size_t N>
using FloatN = Tuple<float, N>;

template <size_t N>
using DoubleN = Tuple<double, N>;

template <size_t N>
using SSizeN = Tuple<ssize_t, N>;

template <size_t N>
using SizeN = Tuple<size_t, N>;

//

template <typename T>
class Tuple<T, 1> {
 public:
    T x;

    constexpr Tuple() : x(T{}) {}

    constexpr Tuple(const T& x_) : x(x_) {}

    T& operator[](size_t i);

    const T& operator[](size_t i) const;
};

template <typename T>
using Tuple1 = Tuple<T, 1>;

using Byte1 = Tuple1<int8_t>;
using UByte1 = Tuple1<uint8_t>;
using Short1 = Tuple1<int16_t>;
using UShort1 = Tuple1<uint16_t>;
using Int1 = Tuple1<int32_t>;
using UInt1 = Tuple1<uint32_t>;
using Long1 = Tuple1<int64_t>;
using ULong1 = Tuple1<uint64_t>;
using Float1 = Tuple1<float>;
using Double1 = Tuple1<double>;
using SSize1 = Tuple1<ssize_t>;
using Size1 = Tuple1<size_t>;

//

template <typename T>
class Tuple<T, 2> {
 public:
    T x;
    T y;

    constexpr Tuple() : x(T{}), y(T{}) {}

    constexpr Tuple(const T& x_, const T& y_) : x(x_), y(y_) {}

    T& operator[](size_t i);

    const T& operator[](size_t i) const;
};

template <typename T>
using Tuple2 = Tuple<T, 2>;

using Byte2 = Tuple2<int8_t>;
using UByte2 = Tuple2<uint8_t>;
using Short2 = Tuple2<int16_t>;
using UShort2 = Tuple2<uint16_t>;
using Int2 = Tuple2<int32_t>;
using UInt2 = Tuple2<uint32_t>;
using Long2 = Tuple2<int64_t>;
using ULong2 = Tuple2<uint64_t>;
using Float2 = Tuple2<float>;
using Double2 = Tuple2<double>;
using SSize2 = Tuple2<ssize_t>;
using Size2 = Tuple2<size_t>;

//

template <typename T>
class Tuple<T, 3> {
 public:
    T x;
    T y;
    T z;

    constexpr Tuple() : x(T{}), y(T{}), z(T{}) {}

    constexpr Tuple(const T& x_, const T& y_, const T& z_)
        : x(x_), y(y_), z(z_) {}

    T& operator[](size_t i);

    const T& operator[](size_t i) const;
};

template <typename T>
using Tuple3 = Tuple<T, 3>;

using Byte3 = Tuple3<int8_t>;
using UByte3 = Tuple3<uint8_t>;
using Short3 = Tuple3<int16_t>;
using UShort3 = Tuple3<uint16_t>;
using Int3 = Tuple3<int33_t>;
using UInt3 = Tuple3<uint33_t>;
using Long3 = Tuple3<int64_t>;
using ULong3 = Tuple3<uint64_t>;
using Float3 = Tuple3<float>;
using Double3 = Tuple3<double>;
using SSize3 = Tuple3<ssize_t>;
using Size3 = Tuple3<size_t>;

//

template <typename T>
class Tuple<T, 4> {
 public:
    T x;
    T y;
    T z;
    T w;

    constexpr Tuple() : x(T{}), y(T{}), z(T{}), w(T{}) {}

    constexpr Tuple(const T& x_, const T& y_, const T& z_, const T& w_)
        : x(x_), y(y_), z(z_), w(w_) {}

    T& operator[](size_t i);

    const T& operator[](size_t i) const;
};

template <typename T>
using Tuple4 = Tuple<T, 4>;

using Byte4 = Tuple4<int8_t>;
using UByte4 = Tuple4<uint8_t>;
using Short4 = Tuple4<int16_t>;
using UShort4 = Tuple4<uint16_t>;
using Int4 = Tuple4<int32_t>;
using UInt4 = Tuple4<uint32_t>;
using Long4 = Tuple4<int64_t>;
using ULong4 = Tuple4<uint64_t>;
using Float4 = Tuple4<float>;
using Double4 = Tuple4<double>;
using SSize4 = Tuple4<ssize_t>;
using Size4 = Tuple4<size_t>;

}  // namespace jet

#include <jet/detail/tuple-inl.h>
#include <jet/tuple_utils.h>

#endif  // INCLUDE_JET_TUPLE_H_
