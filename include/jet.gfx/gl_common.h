// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_GL_COMMON_H_
#define INCLUDE_JET_GFX_GL_COMMON_H_

#ifdef JET_USE_GL

// This file contains OpenGL type forwardings in order to avoid including GL.h
// header files.

// Base GL types
using GLenum = unsigned int;
using GLboolean = unsigned char;
using GLbitfield = unsigned int;
using GLbyte = signed char;
using GLshort = short;
using GLint = int;
using GLsizei = int;
using GLubyte = unsigned char;
using GLushort = unsigned short;
using GLuint = unsigned int;
using GLhalf = unsigned short;
using GLfloat = float;
using GLclampf = float;
using GLdouble = double;
using GLclampd = double;
using GLvoid = void;

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_GFX_GL_COMMON_H_
