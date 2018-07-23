// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MACROS_H_
#define INCLUDE_JET_MACROS_H_

////////////////////////////////////////////////////////////////////////////////
// MARK: Platforms
#if defined(_WIN32) || defined(_WIN64)
#define JET_WINDOWS
#elif defined(__APPLE__)
#define JET_APPLE
#ifndef JET_IOS
#define JET_MACOSX
#endif
#elif defined(linux) || defined(__linux__)
#define JET_LINUX
#endif

////////////////////////////////////////////////////////////////////////////////
// MARK: Debug mode
#if defined(DEBUG) || defined(_DEBUG)
#define JET_DEBUG_MODE
#include <cassert>
#define JET_ASSERT(x) assert(x)
#else
#define JET_ASSERT(x)
#endif

////////////////////////////////////////////////////////////////////////////////
// MARK: C++ shortcuts
#ifdef __cplusplus
#define JET_NON_COPYABLE(ClassName)       \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete;
#endif

////////////////////////////////////////////////////////////////////////////////
// MARK: C++ exceptions
#ifdef __cplusplus
#include <stdexcept>
#define JET_THROW_INVALID_ARG_IF(expression)      \
    if (expression) {                             \
        throw std::invalid_argument(#expression); \
    }
#define JET_THROW_INVALID_ARG_WITH_MESSAGE_IF(expression, message) \
    if (expression) {                                              \
        throw std::invalid_argument(message);                      \
    }
#endif

////////////////////////////////////////////////////////////////////////////////
// MARK: Windows specific
#ifdef JET_WINDOWS
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

////////////////////////////////////////////////////////////////////////////////
// MARK: CUDA specific
#ifdef JET_USE_CUDA

// Host vs. device
#ifdef __CUDACC__
#define JET_CUDA_DEVICE __device__
#define JET_CUDA_HOST __host__
#else
#define JET_CUDA_DEVICE
#define JET_CUDA_HOST
#endif  // __CUDACC__
#define JET_CUDA_HOST_DEVICE JET_CUDA_HOST JET_CUDA_DEVICE

// Alignment
#ifdef __CUDACC__  // NVCC
#define JET_CUDA_ALIGN(n) __align__(n)
#elif defined(__GNUC__)  // GCC
#define JET_CUDA_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)  // MSVC
#define JET_CUDA_ALIGN(n) __declspec(align(n))
#else
#error "Don't know how to handle JET_CUDA_ALIGN"
#endif  // __CUDACC__

// Exception
#define _JET_CUDA_CHECK(result, msg, file, line)                            \
    if (result != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\" \n", file, \
                line, static_cast<unsigned int>(result),                    \
                cudaGetErrorString(result), msg);                           \
        cudaDeviceReset();                                                  \
        exit(EXIT_FAILURE);                                                 \
    }

#define JET_CUDA_CHECK(expression) \
    _JET_CUDA_CHECK((expression), #expression, __FILE__, __LINE__)

#define JET_CUDA_CHECK_LAST_ERROR(msg) \
    _JET_CUDA_CHECK(cudaGetLastError(), msg, __FILE__, __LINE__)

#endif  // JET_USE_CUDA

////////////////////////////////////////////////////////////////////////////////
// MARK: Compiler-specific warning toggle
#define JET_DIAG_STR(s) #s
#define JET_DIAG_JOINSTR(x, y) JET_DIAG_STR(x##y)
#ifdef _MSC_VER
#define JET_DIAG_DO_PRAGMA(x) __pragma(#x)
#define JET_DIAG_PRAGMA(compiler, x) JET_DIAG_DO_PRAGMA(warning(x))
#else
#define JET_DIAG_DO_PRAGMA(x) _Pragma(#x)
#define JET_DIAG_PRAGMA(compiler, x) JET_DIAG_DO_PRAGMA(compiler diagnostic x)
#endif
#if defined(__clang__)
#define JET_DISABLE_CLANG_WARNING(clang_option) \
    JET_DIAG_PRAGMA(clang, push)                \
    JET_DIAG_PRAGMA(clang, ignored JET_DIAG_JOINSTR(-W, clang_option))
#define JET_ENABLE_CLANG_WARNING(clang_option) JET_DIAG_PRAGMA(clang, pop)
#define JET_DISABLE_MSVC_WARNING(gcc_option)
#define JET_ENABLE_MSVC_WARNING(gcc_option)
#define JET_DISABLE_GCC_WARNING(gcc_option)
#define JET_ENABLE_GCC_WARNING(gcc_option)
#elif defined(_MSC_VER)
#define JET_DISABLE_CLANG_WARNING(gcc_option)
#define JET_ENABLE_CLANG_WARNING(gcc_option)
#define JET_DISABLE_MSVC_WARNING(msvc_errorcode) \
    JET_DIAG_PRAGMA(msvc, push)                  \
    JET_DIAG_DO_PRAGMA(warning(disable :##msvc_errorcode))
#define JET_ENABLE_MSVC_WARNING(msvc_errorcode) JET_DIAG_PRAGMA(msvc, pop)
#define JET_DISABLE_GCC_WARNING(gcc_option)
#define JET_ENABLE_GCC_WARNING(gcc_option)
#elif defined(__GNUC__)
#if ((__GNUC__ * 100) + __GNUC_MINOR__) >= 406
#define JET_DISABLE_CLANG_WARNING(gcc_option)
#define JET_ENABLE_CLANG_WARNING(gcc_option)
#define JET_DISABLE_MSVC_WARNING(gcc_option)
#define JET_ENABLE_MSVC_WARNING(gcc_option)
#define JET_DISABLE_GCC_WARNING(gcc_option) \
    JET_DIAG_PRAGMA(GCC, push)              \
    JET_DIAG_PRAGMA(GCC, ignored JET_DIAG_JOINSTR(-W, gcc_option))
#define JET_ENABLE_GCC_WARNING(gcc_option) JET_DIAG_PRAGMA(GCC, pop)
#else
#define JET_DISABLE_CLANG_WARNING(gcc_option)
#define JET_ENABLE_CLANG_WARNING(gcc_option)
#define JET_DISABLE_MSVC_WARNING(gcc_option)
#define JET_ENABLE_MSVC_WARNING(gcc_option)
#define JET_DISABLE_GCC_WARNING(gcc_option) \
    JET_DIAG_PRAGMA(GCC, ignored JET_DIAG_JOINSTR(-W, gcc_option))
#define JET_ENABLE_GCC_WARNING(gcc_option) \
    JET_DIAG_PRAGMA(GCC, warning JET_DIAG_JOINSTR(-W, gcc_option))
#endif
#endif

#endif  // INCLUDE_JET_MACROS_H_
