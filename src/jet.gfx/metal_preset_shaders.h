// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_GFX_METAL_PRESET_SHADERS_H_
#define SRC_JET_GFX_METAL_PRESET_SHADERS_H_

namespace jet {
namespace gfx {

const char kSimpleColorShader[] = R"metal(
    #include <metal_stdlib>
    using namespace metal;

    struct VertexIn {
        float3 position [[attribute(0)]];
        float4 color    [[attribute(1)]];
    };

    struct VertexOut {
        float4 position [[position]];
        float4 color;
    };

    vertex VertexOut vertFunc(VertexIn vertexIn [[stage_in]]) {
        VertexOut vertexOut;
        vertexOut.position = float4(vertexIn.position, 1.0);
        vertexOut.color = vertexIn.color;
        return vertexOut;
    }

    fragment float4 fragFunc(VertexOut fragData [[stage_in]]) {
        return fragData.color;
    }
)metal";

const char kPointsShaders[] = R"metal(
    #include <metal_stdlib>
    using namespace metal;

    struct VertexIn {
        float3 position [[attribute(0)]];
        float4 color    [[attribute(1)]];
    };

    struct VertexOut {
        float4 position [[position]];
        float4 color;
        float pointSize [[point_size]];
    };

    vertex VertexOut vertFunc(VertexIn vertexIn [[stage_in]]) {
        VertexOut vertexOut;
        vertexOut.position = float4(vertexIn.position, 1.0);
        vertexOut.color = vertexIn.color;
        vertexOut.pointSize = 30.0;  // TODO: Test code
        return vertexOut;
    }

    fragment float4 fragFunc(VertexOut fragData [[stage_in]],
                             float2 pointCoord [[point_coord]]) {
        if (length(pointCoord - float2(0.5)) > 0.5) {
            discard_fragment();
        }
        return fragData.color;
    }
)metal";

}  // namespace gfx
}  // namespace jet

#endif  // SRC_JET_GFX_METAL_PRESET_SHADERS_H_
