cmake_minimum_required(VERSION 3.20)
project(essentia_worker LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ESSENTIA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/essentia" CACHE PATH "Pinned upstream sources")
set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/eigen-3.4.0" CACHE PATH "Eigen headers")
set(TENSORFLOW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/libtensorflow" CACHE PATH "TensorFlow C API")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ESSENTIA_REVISION "" CACHE STRING "Verified archive revision, or empty for a Git checkout")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"
  "${ESSENTIA_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" ${ESSENTIA_REVISION}
  RESULT_VARIABLE generation_status COMMAND_ERROR_IS_FATAL ANY)
include("${CMAKE_CURRENT_BINARY_DIR}/algorithms.cmake")
set(SRC "${ESSENTIA_ROOT}/src")
file(GLOB_RECURSE CORE CONFIGURE_DEPENDS "${SRC}/essentia/*.cpp")
list(FILTER CORE EXCLUDE REGEX "(audiocontext|yamlast|jsonconvert|extractor_music|extractor_freesound|ringbuffer)")
list(REMOVE_ITEM CORE "${SRC}/essentia/essentia.cpp")
list(APPEND CORE "${CMAKE_CURRENT_BINARY_DIR}/essentia.cpp")
add_library(essentia_native STATIC ${CORE} ${ALGORITHM_SOURCES}
  "${CMAKE_CURRENT_BINARY_DIR}/registry.cpp"
  "${SRC}/3rdparty/kiss_fft130/kiss_fft.c" "${SRC}/3rdparty/kiss_fft130/tools/kiss_fftr.c")
target_include_directories(essentia_native PUBLIC
  "${CMAKE_CURRENT_BINARY_DIR}" "${SRC}" "${SRC}/essentia" "${SRC}/essentia/scheduler"
  "${SRC}/essentia/streaming" "${SRC}/essentia/streaming/algorithms" "${SRC}/essentia/utils"
  "${TENSORFLOW_ROOT}/include" "${SRC}/3rdparty" "${SRC}/3rdparty/kiss_fft130" "${EIGEN_ROOT}")
target_compile_definitions(essentia_native PUBLIC NOMINMAX EIGEN_MPL2_ONLY)
if(MSVC)
  target_compile_options(essentia_native PUBLIC /EHsc /Zc:__cplusplus /permissive- /bigobj /utf-8)
endif()
if(MINGW)
  target_compile_options(essentia_native PUBLIC -fms-extensions -fdelayed-template-parsing)
  target_link_options(essentia_native PUBLIC -static)
endif()
find_library(TENSORFLOW_LIBRARY NAMES tensorflow PATHS "${TENSORFLOW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(essentia_native PUBLIC "${TENSORFLOW_LIBRARY}")
add_executable(essentia-worker worker.cpp)
target_link_libraries(essentia-worker PRIVATE essentia_native)
if(WIN32)
  add_custom_command(TARGET essentia-worker POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${TENSORFLOW_ROOT}/lib/tensorflow.dll" "$<TARGET_FILE_DIR:essentia-worker>/tensorflow.dll")
endif()
enable_testing()
add_test(NAME native_self_test COMMAND essentia-worker --self-test)
