
#
# Platform and architecture setup
#

option(JET_WARNINGS_AS_ERRORS "Treat all warnings as errors" ON)
if(JET_WARNINGS_AS_ERRORS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(WARN_AS_ERROR_FLAGS "/WX")
    else()
        set(WARN_AS_ERROR_FLAGS "-Werror")
    endif()
endif()

# Get upper case system name
string(TOUPPER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_UPPER)

# Determine architecture (32/64 bit)
set(X64 OFF)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(X64 ON)
endif()


#
# Project options
#

set(DEFAULT_PROJECT_OPTIONS
    CXX_STANDARD              11 # Not available before CMake 3.1; see below for manual command line argument addition
    LINKER_LANGUAGE           "CXX"
    POSITION_INDEPENDENT_CODE ON
)


#
# Include directories
#

set(DEFAULT_INCLUDE_DIRECTORIES)


#
# Libraries
#

set(DEFAULT_LIBRARIES
  PUBLIC
  ${TASKING_SYSTEM_LIBS}
  PRIVATE
)


#
# Compile definitions
#

set(DEFAULT_COMPILE_DEFINITIONS
    SYSTEM_${SYSTEM_NAME_UPPER}
)

# MSVC compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(DEFAULT_COMPILE_DEFINITIONS ${DEFAULT_COMPILE_DEFINITIONS}
        _SCL_SECURE_NO_WARNINGS  # Calling any one of the potentially unsafe methods in the Standard C++ Library
        _CRT_SECURE_NO_WARNINGS  # Calling any one of the potentially unsafe methods in the CRT Library
    )
endif ()


#
# Compile options
#

set(DEFAULT_COMPILE_OPTIONS)

# MSVC compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
        /MP           # -> build with multiple processes
        /W4           # -> warning level 4
        ${WARN_AS_ERROR_FLAGS}

        # /wd4251       # -> disable warning: 'identifier': class 'type' needs to have dll-interface to be used by clients of class 'type2'
        # /wd4592       # -> disable warning: 'identifier': symbol will be dynamically initialized (implementation limitation)
        # /wd4201     # -> disable warning: nonstandard extension used: nameless struct/union (caused by GLM)
        # /wd4127     # -> disable warning: conditional expression is constant (caused by Qt)
        /wd4717 # -> disable warning:  recursive on all control paths, function will cause runtime stack overflow (wrong warning)
        /wd4819 # -> disable warning:  The file contains a character that cannot be represented in the current code page (949) (wrong warning)

        #$<$<CONFIG:Debug>:
        #/RTCc         # -> value is assigned to a smaller data type and results in a data loss
        #>

        $<$<CONFIG:Release>:
        /Gw           # -> whole program global optimization
        /GS-          # -> buffer security check: no
        /GL           # -> whole program optimization: enable link-time code generation (disables Zi)
        /GF           # -> enable string pooling
        >

        # No manual c++11 enable for MSVC as all supported MSVC versions for cmake-init have C++11 implicitly enabled (MSVC >=2013)
    )
endif ()

# GCC and Clang compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
        -Wall
        ${WARN_AS_ERROR_FLAGS}

        # Required for CMake < 3.1; should be removed if minimum required CMake version is raised.
        $<$<VERSION_LESS:${CMAKE_VERSION},3.1>:
            -std=c++11
        >
    )
endif ()


#
# Linker options
#

set(DEFAULT_LINKER_OPTIONS)

# Use pthreads on mingw and linux
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(DEFAULT_LINKER_OPTIONS
        -pthread
    )
endif()

# Code coverage - Debug only
# NOTE: Code coverage results with an optimized (non-Debug) build may be misleading
if (CMAKE_BUILD_TYPE MATCHES Debug AND CMAKE_COMPILER_IS_GNUCXX)
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
        -g
        -O0
        -fprofile-arcs
        -ftest-coverage
    )

    set(DEFAULT_LINKER_OPTIONS ${DEFAULT_LINKER_OPTIONS}
        -fprofile-arcs
        -ftest-coverage
    )
endif()
