// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#pragma once

#ifndef UNUSED_VARIABLE
#   define UNUSED_VARIABLE(x) ((void)x)
#endif

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
#define NOMINMAX
#include <Windows.h>
#include <exception>
#include <string>

#include <jet/macros.h>
#include "macros.h"

inline void throwIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        throw std::exception(std::to_string(hr).c_str());
    }
}

#ifndef IF_FAILED_CLEANUP
#   define IF_FAILED_CLEANUP(_hr) if (FAILED(_hr)) { hr = _hr; goto Cleanup; }
#endif

#ifndef FAIL_AND_CLEANUP
#   define FAIL_AND_CLEANUP(_hr) { hr = _hr; goto Cleanup; }
#endif
