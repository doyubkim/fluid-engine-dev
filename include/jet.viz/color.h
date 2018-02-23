// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_COLOR_H_
#define INCLUDE_JET_VIZ_COLOR_H_

#include <jet/vector4.h>
#include <cstdint>

namespace jet {
namespace viz {

struct Color;

//! Simple 8-bit RGBA color representation.
struct ByteColor final {
    //! Red channel value.
    uint8_t r = 0;

    //! Green channel value.
    uint8_t g = 0;

    //! Blue channel value.
    uint8_t b = 0;

    //! Alpha channel value.
    uint8_t a = 0;

    //! Default constructor.
    ByteColor();

    //! Constructs 8-bit color with 32-bit float color.
    explicit ByteColor(const Color& other);

    //! Constructs 8-bit color with given r, g, b, and a values.
    ByteColor(uint8_t newR, uint8_t newG, uint8_t newB, uint8_t newA);

    //! Copy constructor.
    ByteColor(const ByteColor& other);

    //! Makes white color.
    static ByteColor makeWhite();

    //! Makes black color.
    static ByteColor makeBlack();

    //! Makes red color.
    static ByteColor makeRed();

    //! Makes green color.
    static ByteColor makeGreen();

    //! Makes blue color.
    static ByteColor makeBlue();

    //!
    //! Makes color with jet colormap.
    //!
    //! \param value Input scalar value in [-1, 1] range.
    //! \return New color instance.
    //!
    static ByteColor makeJet(float value);
};

//! Simple 32-bit floating point RGBA color representation.
struct Color final {
    //! Red channel value.
    float r = 0.f;

    //! Green channel value.
    float g = 0.f;

    //! Blue channel value.
    float b = 0.f;

    //! Alpha channel value.
    float a = 0.f;

    //! Default constructor.
    Color();

    //! Constructs 32-bit float color with 8-bit color.
    explicit Color(const ByteColor& other);

    //! Constructs color from 4-D vector.
    explicit Color(const Vector4F& rgba);

    //! Constructs color from 3-D vector and alpha value.
    explicit Color(const Vector3F& rgb, float alpha);

    //! Constructs color with r, g, b, and a values.
    explicit Color(float newR, float newG, float newB, float newA);

    //! Copy constructor.
    Color(const Color& other);

    //! Makes white color.
    static Color makeWhite();

    //! Makes black color.
    static Color makeBlack();

    //! Makes red color.
    static Color makeRed();

    //! Makes green color.
    static Color makeGreen();

    //! Makes blue color.
    static Color makeBlue();

    //!
    //! Makes color with jet colormap.
    //!
    //! \param value Input scalar value in [-1, 1] range.
    //! \return New color instance.
    //!
    static Color makeJet(float value);
};

}  // namespace viz
}  // namespace jet

#include "detail/color-inl.h"

#endif  // INCLUDE_JET_VIZ_COLOR_H_
