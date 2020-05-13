// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

// Jet public headers
#include <jet/logging.h>
#include <jet/macros.h>

#ifdef JET_WINDOWS
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <objbase.h>
#endif

// Jet private headers
#include <private_helpers.h>
