// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_DETAIL_COLOR_INL_H_
#define INCLUDE_JET_VIZ_DETAIL_COLOR_INL_H_

#include <jet/math_utils.h>

namespace jet {
namespace viz {

inline ByteColor::ByteColor() {}

inline ByteColor::ByteColor(const Color& other) {
    r = static_cast<uint8_t>(clamp(other.r, 0.f, 1.f) * 255.f);
    g = static_cast<uint8_t>(clamp(other.g, 0.f, 1.f) * 255.f);
    b = static_cast<uint8_t>(clamp(other.b, 0.f, 1.f) * 255.f);
    a = static_cast<uint8_t>(clamp(other.a, 0.f, 1.f) * 255.f);
}

inline ByteColor::ByteColor(uint8_t newR, uint8_t newG, uint8_t newB,
                            uint8_t newA)
    : r(newR), g(newG), b(newB), a(newA) {}

inline ByteColor::ByteColor(const ByteColor& other)
    : r(other.r), g(other.g), b(other.b), a(other.a) {}

inline ByteColor ByteColor::makeWhite() {
    return ByteColor(255, 255, 255, 255);
}

inline ByteColor ByteColor::makeBlack() { return ByteColor(0, 0, 0, 255); }

inline ByteColor ByteColor::makeRed() { return ByteColor(255, 0, 0, 255); }

inline ByteColor ByteColor::makeGreen() { return ByteColor(0, 255, 0, 255); }

inline ByteColor ByteColor::makeBlue() { return ByteColor(0, 0, 255, 255); }

inline ByteColor ByteColor::makeJet(float value) {
    return ByteColor(Color::makeJet(value));
}

inline Color::Color() {}

inline Color::Color(const ByteColor& other) {
    r = static_cast<float>(other.r) / 255.f;
    g = static_cast<float>(other.g) / 255.f;
    b = static_cast<float>(other.b) / 255.f;
    a = static_cast<float>(other.a) / 255.f;
}

inline Color::Color(float newR, float newG, float newB, float newA)
    : r(newR), g(newG), b(newB), a(newA) {}

inline Color::Color(const Vector4F& vec4) {
    r = clamp(vec4.x, 0.f, 1.f);
    g = clamp(vec4.y, 0.f, 1.f);
    b = clamp(vec4.z, 0.f, 1.f);
    a = clamp(vec4.w, 0.f, 1.f);
}

inline Color::Color(const Vector3F& vec3, float alpha) {
    r = clamp(vec3.x, 0.f, 1.f);
    g = clamp(vec3.y, 0.f, 1.f);
    b = clamp(vec3.z, 0.f, 1.f);
    a = clamp(alpha, 0.f, 1.f);
}

inline Color::Color(const Color& other)
    : r(other.r), g(other.g), b(other.b), a(other.a) {}

inline Color Color::makeWhite() { return Color(1.f, 1.f, 1.f, 1.f); }

inline Color Color::makeBlack() { return Color(0.f, 0.f, 0.f, 1.f); }

inline Color Color::makeRed() { return Color(1.f, 0.f, 0.f, 1.f); }

inline Color Color::makeGreen() { return Color(0.f, 1.f, 0.f, 1.f); }

inline Color Color::makeBlue() { return Color(0.f, 0.f, 1.f, 1.f); }

inline Color Color::makeJet(float value) {
    // Adopted from
    // https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
    static const auto interpolate = [](float val, float y0, float x0, float y1,
                                       float x1) -> float {
        return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
    };

    static const auto base = [](float val) -> float {
        if (val <= -0.75) {
            return 0;
        } else if (val <= -0.25f) {
            return interpolate(val, 0.0f, -0.75f, 1.0f, -0.25f);
        } else if (val <= 0.25f) {
            return 1.0f;
        } else if (val <= 0.75f) {
            return interpolate(val, 1.0f, 0.25f, 0.0f, 0.75f);
        } else {
            return 0.0f;
        }
    };

    static const auto red = [](float val) { return base(val - 0.5f); };
    static const auto green = [](float val) { return base(val); };
    static const auto blue = [](float val) { return base(val + 0.5f); };

    return Color{red(value), green(value), blue(value), 1.0f};
}

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_DETAIL_COLOR_INL_H_
