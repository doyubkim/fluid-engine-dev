// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_RENDER_STATES_H_
#define INCLUDE_JET_VIZ_RENDER_STATES_H_

namespace jet {
namespace viz {

//! Rendering state representation.
struct RenderStates final {
    //! Cull modes.
    enum class CullMode {
        //! No culling.
        None = 0,

        //! Front-face culling.
        Front,

        //! Back-face culling.
        Back,
    };

    //! Alpha-blending factors.
    enum class BlendFactor {
        //! Use 0 blend factor.
        Zero = 0,

        //! Use 1 blend factor.
        One,

        //! Use source color blend factor.
        SrcColor,

        //! Use 1 - source color blend factor.
        OneMinusSrcColor,

        //! Use source alpha blend factor.
        SrcAlpha,

        //! Use 1 - source alpha blend factor.
        OneMinusSrcAlpha,

        //! Use destination alpha blend factor.
        DestAlpha,

        //! Use 1 - destination alpha blend factor.
        OneMinusDestAlpha,

        //! Use destination color blend factor.
        DestColor,

        //! Use 1 - destination color blend factor.
        OneMinusDestColor,
    };

    //! True if front face is defined as clock-wise order.
    bool isFrontFaceClockWise = true;

    //! True if blending is enabled.
    bool isBlendEnabled = true;

    //! True if depth test is enabled.
    bool isDepthTestEnabled = true;

    //! The cull mode.
    CullMode cullMode = CullMode::Back;

    //! The blend factor for the source.
    BlendFactor sourceBlendFactor = BlendFactor::SrcAlpha;

    //! The blend factor for the destination.
    BlendFactor destinationBlendFactor = BlendFactor::OneMinusSrcAlpha;
};

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_RENDER_STATES_H_
