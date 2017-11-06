// Copyright (c) 2017 Doyub Kim
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

struct ByteColor final {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 0;

    ByteColor();

    explicit ByteColor(const Color& other);

    explicit ByteColor(uint8_t newR, uint8_t newG, uint8_t newB, uint8_t newA);

    ByteColor(const ByteColor& other);

    static ByteColor makeWhite();

    static ByteColor makeBlack();

    static ByteColor makeRed();

    static ByteColor makeGreen();

    static ByteColor makeBlue();
};

struct Color final {
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
    float a = 0.f;

    Color();

    explicit Color(const ByteColor& other);

    explicit Color(const Vector4F& rgba);

    explicit Color(const Vector3F& rgb, float alpha);

    explicit Color(float newR, float newG, float newB, float newA);

    Color(const Color& other);

    static Color makeWhite();

    static Color makeBlack();

    static Color makeRed();

    static Color makeGreen();

    static Color makeBlue();

    static Color makeJet(float value);
};

}  // namespace viz
}  // namespace jet

#include "detail/color-inl.h"

#endif  // INCLUDE_JET_VIZ_COLOR_H_
