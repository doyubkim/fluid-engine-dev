// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/color_utils.h>

namespace jet {
namespace gfx {

namespace {

inline float interpolate(float val, float y0, float x0, float y1, float x1) {
    return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
};

inline float jetBase(float val) {
    if (val <= -0.75f) {
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

inline float jetRed(float value) { return jetBase(value - 0.5f); }

inline float jetGreen(float value) { return jetBase(value); }

inline float jetBlue(float value) { return jetBase(value + 0.5f); }

}  // namespace

Vector4F ColorUtils::makeWhite() { return Vector4F(1, 1, 1, 1); }

Vector4F ColorUtils::makeGray() { return Vector4F(0.5f, 0.5f, 0.5f, 1); }

Vector4F ColorUtils::makeBlack() { return Vector4F(0, 0, 0, 1); }

Vector4F ColorUtils::makeRed() { return Vector4F(1, 0, 0, 1); }

Vector4F ColorUtils::makeGreen() { return Vector4F(0, 1, 0, 1); }

Vector4F ColorUtils::makeBlue() { return Vector4F(0, 0, 1, 1); }

Vector4F ColorUtils::makeCyan() { return Vector4F(0, 1, 1, 1); }

Vector4F ColorUtils::makeMagenta() { return Vector4F(1, 0, 1, 1); }

Vector4F ColorUtils::makeYellow() { return Vector4F(1, 1, 0, 1); }

Vector4F ColorUtils::makeJet(float value) {
    // Adopted from
    // https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
    return Vector4F(jetRed(value), jetGreen(value), jetBlue(value), 1.0f);
}

}  // namespace gfx
}  // namespace jet