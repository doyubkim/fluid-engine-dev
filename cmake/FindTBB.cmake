
#
# Find TBB
#

set(TBB_VERSION_REQUIRED "2.0")

if (NOT TBB_ROOT)
  set(TBB_ROOT $ENV{TBB_ROOT})
endif()
if (NOT TBB_ROOT)
  set(TBB_ROOT $ENV{TBBROOT})
endif()

# detect changed TBB_ROOT
if (NOT TBB_ROOT STREQUAL TBB_ROOT_LAST)
  unset(TBB_INCLUDE_DIR CACHE)
  unset(TBB_LIBRARY CACHE)
  unset(TBB_LIBRARY_DEBUG CACHE)
  unset(TBB_LIBRARY_MALLOC CACHE)
  unset(TBB_LIBRARY_MALLOC_DEBUG CACHE)
  unset(TBB_INCLUDE_DIR_MIC CACHE)
  unset(TBB_LIBRARY_MIC CACHE)
  unset(TBB_LIBRARY_MALLOC_MIC CACHE)
endif()

if (WIN32)
  # workaround for parentheses in variable name / CMP0053
  set(PROGRAMFILESx86 "PROGRAMFILES(x86)")
  set(PROGRAMFILES32 "$ENV{${PROGRAMFILESx86}}")
  if (NOT PROGRAMFILES32)
    set(PROGRAMFILES32 "$ENV{PROGRAMFILES}")
  endif()
  if (NOT PROGRAMFILES32)
    set(PROGRAMFILES32 "C:/Program Files (x86)")
  endif()
  find_path(TBB_ROOT include/tbb/task_scheduler_init.h
    DOC "Root of TBB installation"
    HINTS ${TBB_ROOT}
    PATHS
      ${PROJECT_SOURCE_DIR}/tbb
      ${PROJECT_SOURCE_DIR}/../tbb
      "${PROGRAMFILES32}/IntelSWTools/compilers_and_libraries/windows/tbb"
      "${PROGRAMFILES32}/Intel/Composer XE/tbb"
      "${PROGRAMFILES32}/Intel/compilers_and_libraries/windows/tbb"
  )

  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(TBB_ARCH intel64)
  else()
    set(TBB_ARCH ia32)
  endif()

  if (MSVC10)
    set(TBB_VCVER vc10)
  elseif (MSVC11)
    set(TBB_VCVER vc11)
  elseif (MSVC12)
    set(TBB_VCVER vc12)
  else()
    set(TBB_VCVER vc14)
  endif()

  set(TBB_LIBDIR ${TBB_ROOT}/lib/${TBB_ARCH}/${TBB_VCVER})
  set(TBB_BINDIR ${TBB_ROOT}/bin/${TBB_ARCH}/${TBB_VCVER})

  find_path(TBB_INCLUDE_DIR tbb/task_scheduler_init.h PATHS ${TBB_ROOT}/include NO_DEFAULT_PATH)
  find_library(TBB_LIBRARY tbb PATHS ${TBB_LIBDIR} NO_DEFAULT_PATH)
  find_library(TBB_LIBRARY_DEBUG tbb_debug PATHS ${TBB_LIBDIR} NO_DEFAULT_PATH)
  find_library(TBB_LIBRARY_MALLOC tbbmalloc PATHS ${TBB_LIBDIR} NO_DEFAULT_PATH)
  find_library(TBB_LIBRARY_MALLOC_DEBUG tbbmalloc_debug PATHS ${TBB_LIBDIR} NO_DEFAULT_PATH)

else ()

  find_path(TBB_ROOT include/tbb/task_scheduler_init.h
    DOC "Root of TBB installation"
    HINTS ${TBB_ROOT}
    PATHS
      ${PROJECT_SOURCE_DIR}/tbb
      /opt/intel/composerxe/tbb
      /opt/intel/compilers_and_libraries/tbb
  )

  if (APPLE)
    find_path(TBB_INCLUDE_DIR tbb/task_scheduler_init.h PATHS ${TBB_ROOT}/include NO_DEFAULT_PATH)
    find_library(TBB_LIBRARY tbb PATHS ${TBB_ROOT}/lib NO_DEFAULT_PATH)
    find_library(TBB_LIBRARY_DEBUG tbb_debug PATHS ${TBB_ROOT}/lib NO_DEFAULT_PATH)
    find_library(TBB_LIBRARY_MALLOC tbbmalloc PATHS ${TBB_ROOT}/lib NO_DEFAULT_PATH)
    find_library(TBB_LIBRARY_MALLOC_DEBUG tbbmalloc_debug PATHS ${TBB_ROOT}/lib NO_DEFAULT_PATH)
  else()
    find_path(TBB_INCLUDE_DIR tbb/task_scheduler_init.h PATHS ${TBB_ROOT}/include NO_DEFAULT_PATH)
    find_library(TBB_LIBRARY libtbb.so.2 HINTS ${TBB_ROOT}/lib/intel64/gcc4.4)
    find_library(TBB_LIBRARY_DEBUG libtbb_debug.so.2 HINTS ${TBB_ROOT}/lib/intel64/gcc4.4)
    find_library(TBB_LIBRARY_MALLOC libtbbmalloc.so.2 HINTS ${TBB_ROOT}/lib/intel64/gcc4.4)
    find_library(TBB_LIBRARY_MALLOC_DEBUG libtbbmalloc_debug.so.2 HINTS ${TBB_ROOT}/lib/intel64/gcc4.4)
  endif()

  find_path(TBB_INCLUDE_DIR_MIC tbb/task_scheduler_init.h PATHS ${TBB_ROOT}/include NO_DEFAULT_PATH)
  find_library(TBB_LIBRARY_MIC libtbb.so.2 PATHS ${TBB_ROOT}/lib/mic NO_DEFAULT_PATH)
  find_library(TBB_LIBRARY_MALLOC_MIC libtbbmalloc.so.2 PATHS ${TBB_ROOT}/lib/mic NO_DEFAULT_PATH)

  mark_as_advanced(TBB_INCLUDE_DIR_MIC)
  mark_as_advanced(TBB_LIBRARY_MIC)
  mark_as_advanced(TBB_LIBRARY_MALLOC_MIC)
endif()

set(TBB_ROOT_LAST ${TBB_ROOT} CACHE INTERNAL "Last value of TBB_ROOT to detect changes")

set(TBB_ERROR_MESSAGE
  "Threading Building Blocks (TBB) with minimum version ${TBB_VERSION_REQUIRED} not found.
Please make sure you have the TBB headers installed as well (the package is typically named 'libtbb-dev' or 'tbb-devel') and/or hint the location of TBB in TBB_ROOT. Alternatively, you can try to use OpenMP as tasking system by setting JET_TASKING_SYSTEM=OpenMP or C++11 threads using JET_TASKING_SYSTEM=CPP11Threads")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB
  ${TBB_ERROR_MESSAGE}
  TBB_INCLUDE_DIR TBB_LIBRARY TBB_LIBRARY_MALLOC
)

# check version
if (TBB_INCLUDE_DIR)
  file(READ ${TBB_INCLUDE_DIR}/tbb/tbb_stddef.h TBB_STDDEF_H)

  string(REGEX MATCH "#define TBB_VERSION_MAJOR ([0-9])" DUMMY "${TBB_STDDEF_H}")
  set(TBB_VERSION_MAJOR ${CMAKE_MATCH_1})

  string(REGEX MATCH "#define TBB_VERSION_MINOR ([0-9])" DUMMY "${TBB_STDDEF_H}")
  set(TBB_VERSION "${TBB_VERSION_MAJOR}.${CMAKE_MATCH_1}")

  if (TBB_VERSION VERSION_LESS TBB_VERSION_REQUIRED)
    message(FATAL_ERROR ${TBB_ERROR_MESSAGE})
  endif()

  set(TBB_VERSION ${TBB_VERSION} CACHE STRING "TBB Version")
  mark_as_advanced(TBB_VERSION)
endif()

if (TBB_FOUND)
  set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
  # NOTE(jda) - TBB found in CentOS 6/7 package manager does not have debug
  #             versions of the library...silently fall-back to using only the
  #             libraries which we actually found.
  if (NOT TBB_LIBRARY_DEBUG)
    set(TBB_LIBRARIES ${TBB_LIBRARY} ${TBB_LIBRARY_MALLOC})
  else ()
    set(TBB_LIBRARIES
        optimized ${TBB_LIBRARY} optimized ${TBB_LIBRARY_MALLOC}
        debug ${TBB_LIBRARY_DEBUG} debug ${TBB_LIBRARY_MALLOC_DEBUG}
    )
  endif()
endif()

if (TBB_INCLUDE_DIR AND TBB_LIBRARY_MIC AND TBB_LIBRARY_MALLOC_MIC)
  set(TBB_FOUND_MIC TRUE)
  set(TBB_INCLUDE_DIRS_MIC ${TBB_INCLUDE_DIR_MIC})
  set(TBB_LIBRARIES_MIC ${TBB_LIBRARY_MIC} ${TBB_LIBRARY_MALLOC_MIC})
endif()

mark_as_advanced(TBB_INCLUDE_DIR)
mark_as_advanced(TBB_LIBRARY)
mark_as_advanced(TBB_LIBRARY_DEBUG)
mark_as_advanced(TBB_LIBRARY_MALLOC)
mark_as_advanced(TBB_LIBRARY_MALLOC_DEBUG)
