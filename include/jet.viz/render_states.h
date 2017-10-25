// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_RENDER_STATES_H_
#define INCLUDE_JET_VIZ_RENDER_STATES_H_

namespace jet { namespace viz {

struct RenderStates final {
    enum class CullMode {
        None = 0,
        Front,
        Back,
    };

    enum class BlendFactor {
        Zero = 0,
        One,
        SrcColor,
        OneMinusSrcColor,
        SrcAlpha,
        OneMinusSrcAlpha,
        DestAlpha,
        OneMinusDestAlpha,
        DestColor,
        OneMinusDestColor,
    };

    bool isFrontFaceClockWise = true;
    bool isBlendEnabled = true;
    bool isDepthTestEnabled = true;
    CullMode cullMode = CullMode::Back;
    BlendFactor sourceBlendFactor = BlendFactor::SrcAlpha;
    BlendFactor destinationBlendFactor = BlendFactor::OneMinusSrcAlpha;
};

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_RENDER_STATES_H_
