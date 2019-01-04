// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_GFX_GL_PRESET_SHADERS_H_
#define SRC_JET_GFX_GL_PRESET_SHADERS_H_

#ifdef JET_USE_GL

#include <jet.gfx/gl_common.h>

namespace jet {
namespace gfx {

const GLchar* kSimpleColorShaders[2] = {
    // Vertex shader
    R"glsl(
    #version 330 core
    uniform mat4 ModelViewProjection;
    in vec3 position;
    in vec4 color;
    out VertexData {
        vec4 color;
    } outData;
    void main() {
        outData.color = color;
        gl_Position = ModelViewProjection * vec4(position,1.0);
    }
    )glsl",

    // Fragment shader
    R"glsl(
    #version 330 core
    in VertexData {
        vec4 color;
    } inData;
    out vec4 fragColor;
    void main() {
        fragColor = inData.color;
    }
    )glsl"};

const GLchar* kSimpleTexture2Shader[2] = {
    // Vertex shader
    R"glsl(
    #version 330 core
    uniform mat4 ModelViewProjection;
    in vec3 position;
    in vec2 texCoord2;
    out VertexData {
        vec2 texCoord2;
    } outData;
    void main() {
        outData.texCoord2 = texCoord2;
        gl_Position = ModelViewProjection * vec4(position,1.0);
    }
    )glsl",

    // Fragment shader
    R"glsl(
    #version 330 core
    uniform sampler2D tex0;
    uniform vec4 Multiplier;
    in VertexData {
        vec2 texCoord2;
    } inData;
    out vec4 fragColor;
    void main() {
        fragColor = Multiplier * texture(tex0, inData.texCoord2);
    }
    )glsl"};

const GLchar* kSimpleTexture3Shader[2] = {
    // Vertex shader
    R"glsl(
    #version 330 core
    uniform mat4 ModelViewProjection;
    in vec3 position;
    in vec3 texCoord3;
    out VertexData {
        vec3 texCoord3;
    } outData;
    void main() {
        outData.texCoord3 = texCoord3;
        gl_Position = ModelViewProjection * vec4(position,1.0);
    }
    )glsl",

    // Fragment shader
    R"glsl(
    #version 330 core
    uniform sampler3D tex0;
    uniform vec4 Multiplier;
    in VertexData {
        vec3 texCoord3;
    } inData;
    out vec4 fragColor;
    void main() {
        fragColor = Multiplier * texture(tex0, inData.texCoord3);
    }
    )glsl"};

const GLchar* kPointsShaders[2] = {
    // Vertex shader
    R"glsl(
    #version 330 core
    uniform mat4 ModelViewProjection;
    uniform float Radius;
    in vec3 position;
    in vec4 color;
    out VertexData {
        vec4 color;
    } outData;
    void main() {
        outData.color = color;
        gl_PointSize = 2.0 * Radius;
        gl_Position = ModelViewProjection * vec4(position,1.0);
    }
    )glsl",

    // Fragment shader
    R"glsl(
    #version 330 core
    uniform float Radius;
    in VertexData {
    	 vec4 color;
    } inData;
    out vec4 fragColor;
    void main() {
         if (length(gl_PointCoord - vec2(0.5, 0.5)) > 0.5) {
             discard;
         }
    	 fragColor = inData.color;
    }
    )glsl"};

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL

#endif  // SRC_JET_GFX_GL_PRESET_SHADERS_H_
