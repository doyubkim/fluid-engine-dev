
#
# Setup tasking system build configuration
#

# Determine the best default option for tasking system backend
if(NOT DEFINED JET_TASKING_SYSTEM)
  find_package(TBB QUIET)
  if(TBB_FOUND)
    set(TASKING_DEFAULT TBB)
  else()
    find_package(OpenMP QUIET)
    if(OpenMP_FOUND)
      set(TASKING_DEFAULT OpenMP)
    else()
      set(TASKING_DEFAULT CPP11Threads)
    endif()
  endif()
else()
  set(TASKING_DEFAULT ${JET_TASKING_SYSTEM})
endif()

set(JET_TASKING_SYSTEM ${TASKING_DEFAULT} CACHE STRING
    "Per-node thread tasking system [CPP11Threads,TBB,OpenMP,Serial]")

set_property(CACHE JET_TASKING_SYSTEM PROPERTY
             STRINGS CPP11Threads TBB OpenMP Serial)

# NOTE(jda) - Make the JET_TASKING_SYSTEM build option case-insensitive
string(TOUPPER ${JET_TASKING_SYSTEM} JET_TASKING_SYSTEM_ID)

set(JET_TASKING_TBB          FALSE)
set(JET_TASKING_OPENMP       FALSE)
set(JET_TASKING_CPP11THREADS FALSE)
set(JET_TASKING_SERIAL       FALSE)

if(${JET_TASKING_SYSTEM_ID} STREQUAL "TBB")
  set(JET_TASKING_TBB TRUE)
elseif(${JET_TASKING_SYSTEM_ID} STREQUAL "OPENMP")
  set(JET_TASKING_OPENMP TRUE)
elseif(${JET_TASKING_SYSTEM_ID} STREQUAL "CPP11THREADS")
  set(JET_TASKING_CPP11THREADS TRUE)
else()
  set(JET_TASKING_SERIAL TRUE)
endif()

unset(TASKING_SYSTEM_LIBS)
unset(TASKING_SYSTEM_LIBS_MIC)

if(JET_TASKING_TBB)
  find_package(TBB REQUIRED)
  add_definitions(-DJET_TASKING_TBB)
  include_directories(${TBB_INCLUDE_DIRS})
  set(TASKING_SYSTEM_LIBS ${TBB_LIBRARIES})
  set(TASKING_SYSTEM_LIBS_MIC ${TBB_LIBRARIES_MIC})
else()
  unset(TBB_INCLUDE_DIR          CACHE)
  unset(TBB_LIBRARY              CACHE)
  unset(TBB_LIBRARY_DEBUG        CACHE)
  unset(TBB_LIBRARY_MALLOC       CACHE)
  unset(TBB_LIBRARY_MALLOC_DEBUG CACHE)
  unset(TBB_INCLUDE_DIR_MIC      CACHE)
  unset(TBB_LIBRARY_MIC          CACHE)
  unset(TBB_LIBRARY_MALLOC_MIC   CACHE)
  if(JET_TASKING_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      add_definitions(-DJET_TASKING_OPENMP)
    endif()
  elseif(JET_TASKING_CPP11THREADS)
      add_definitions(-DJET_TASKING_CPP11THREADS)
  else()#Serial
    # Do nothing, will fall back to scalar code (useful for debugging)
  endif()
endif()
