// Copyright (c) 2016 Doyub Kim

// Jet public headers
#include <jet/logging.h>
#include <jet/macros.h>

#ifdef JET_WINDOWS
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <objbase.h>
#endif

// Jet private headers
#include <private_helpers.h>
