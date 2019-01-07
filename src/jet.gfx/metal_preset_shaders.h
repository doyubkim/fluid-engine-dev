// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_GFX_METAL_PRESET_SHADERS_H_
#define SRC_JET_GFX_METAL_PRESET_SHADERS_H_

#import <simd/types.h>

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

    struct VertexUniforms {
        float4x4 ModelViewProjection;
    };

    vertex VertexOut vertFunc(VertexIn vertexIn [[stage_in]],
                              constant VertexUniforms &uniforms [[buffer(1)]]) {
        VertexOut vertexOut;
        vertexOut.position = uniforms.ModelViewProjection * float4(vertexIn.position, 1.0);
        vertexOut.color = vertexIn.color;
        return vertexOut;
    }

    fragment float4 fragFunc(VertexOut fragData [[stage_in]]) {
        return fragData.color;
    }
)metal";

struct SimpleColorVertexUniforms {
    simd_float4x4 ModelViewProjection;
};

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

    struct VertexUniforms {
        float4x4 ModelViewProjection;
        float Radius;
    };

    vertex VertexOut vertFunc(VertexIn vertexIn [[stage_in]],
                              constant VertexUniforms &uniforms [[buffer(1)]]) {
        VertexOut vertexOut;
        vertexOut.position = uniforms.ModelViewProjection * float4(vertexIn.position, 1.0);
        vertexOut.color = vertexIn.color;
        vertexOut.pointSize = 2.0 * uniforms.Radius;
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

struct PointsVertexUniforms {
    simd_float4x4 ModelViewProjection;
    float Radius;
};

constexpr size_t kDefaultParameterSize = sizeof(simd_float4x4);

}  // namespace gfx
}  // namespace jet

#endif  // SRC_JET_GFX_METAL_PRESET_SHADERS_H_
