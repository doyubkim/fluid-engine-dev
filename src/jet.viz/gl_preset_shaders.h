// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_VIZ_GL_GL_PRESET_SHADERS_H_
#define SRC_JET_VIZ_GL_GL_PRESET_SHADERS_H_

#ifdef JET_USE_GL

namespace jet {
namespace viz {

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
        gl_PointSize = Radius;
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

const GLchar* kPointSpriteShaders[3] = {
    // Vertex shader
    R"glsl(
    #version 330 core
    uniform mat4 ModelViewProjection;
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec4 color;
    out VertexData {
        vec4 color;
    } outData;
    void main() {
        outData.color = color;
        gl_Position = ModelViewProjection * vec4(position,1.0);
    }
    )glsl",

    // Geometry shader
    R"glsl(
    #version 330 core
    uniform float ViewWidth;
    uniform float ViewHeight;
    uniform float Radius;
    layout(points) in;
    layout(triangle_strip, max_vertices=4) out;
    in VertexData {
        vec4 color;
    } inData[1];
    out VertexData {
        vec4 color;
        vec2 texCoord2;
    } outData;
    void main() {
        float size = 2.0 * Radius;
        vec4 position = gl_in[0].gl_Position;
        vec4 color = inData[0].color;
        float dx = size / ViewWidth * position.w;
        float dy = size / ViewHeight * position.w;
        gl_Position = position + vec4(dx, -dy, 0, 0);
        outData.color = color;
        outData.texCoord2 = vec2(1,0);
        EmitVertex();
        gl_Position = position + vec4(-dx, -dy, 0, 0);
        outData.color = color;
        outData.texCoord2 = vec2(0,0);
        EmitVertex();
        gl_Position = position + vec4(dx, dy, 0, 0);
        outData.color = color;
        outData.texCoord2 = vec2(1,1);
        EmitVertex();
        gl_Position = position + vec4(-dx, dy, 0, 0);
        outData.color = color;
        outData.texCoord2 = vec2(0,1);
        EmitVertex();
    }
    )glsl",

    // Fragment shader
    R"glsl(
    #version 330 core
    in VertexData {
    	vec4 color;
        vec2 texCoord2;
    } inData;
    out vec4 fragColor;
    void main() {
        float r = distance(inData.texCoord2, vec2(0.5,0.5));
        if (r > 0.5) {
    	     discard;
        }
    	fragColor = inData.color;
    }
    )glsl"};

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // SRC_JET_VIZ_GL_GL_PRESET_SHADERS_H_
