# tools/ominix_diffusion/src/CMakeLists.txt
# Stable-Diffusion 库，链接统一的 ggml target（由根 CMakeLists 编译）

cmake_minimum_required(VERSION 3.14)

# ─── SD 专属选项 ──────────────────────────────────────────────
option(SD_BUILD_EXAMPLES        "sd: build examples"             ON)
option(SD_CUDA                  "sd: cuda backend"               OFF)
option(SD_HIPBLAS               "sd: rocm backend"               OFF)
option(SD_METAL                 "sd: metal backend"              OFF)
option(SD_VULKAN                "sd: vulkan backend"             OFF)
option(SD_OPENCL                "sd: opencl backend"             OFF)
option(SD_SYCL                  "sd: sycl backend"               OFF)
option(SD_MUSA                  "sd: musa backend"               OFF)
option(SD_CANN                  "sd: cann backend"               ON)
option(SD_BUILD_SHARED_LIBS     "sd: build shared libs"          OFF)

# ─── 后端宏定义 ──────────────────────────────────────────────
if(SD_CUDA)
    add_definitions(-DSD_USE_CUDA)
endif()
if(SD_METAL)
    add_definitions(-DSD_USE_METAL)
endif()
if(SD_VULKAN)
    add_definitions(-DSD_USE_VULKAN)
endif()
if(SD_OPENCL)
    add_definitions(-DSD_USE_OPENCL)
endif()
if(SD_HIPBLAS)
    add_definitions(-DSD_USE_CUDA)
endif()
if(SD_MUSA)
    add_definitions(-DSD_USE_CUDA)
endif()
if(SD_CANN OR GGML_CANN)
    add_definitions(-DSD_USE_CANN)
endif()
if(SD_SYCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -fsycl")
    add_definitions(-DSD_USE_SYCL)
endif()

# ─── 版本信息 ────────────────────────────────────────────────
find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)
if(GIT_EXE)
    execute_process(
        COMMAND ${GIT_EXE} describe --tags --abbrev=7 --dirty=+
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SDCPP_BUILD_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(
        COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SDCPP_BUILD_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT SDCPP_BUILD_VERSION)
    set(SDCPP_BUILD_VERSION "unknown")
endif()
if(NOT SDCPP_BUILD_COMMIT)
    set(SDCPP_BUILD_COMMIT "unknown")
endif()
message(STATUS "[sd] stable-diffusion version ${SDCPP_BUILD_VERSION}")

# ─── 收集源文件 ──────────────────────────────────────────────
file(GLOB SD_LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/vocab/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/vocab/*.cpp"
)

set_property(
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp
    APPEND PROPERTY COMPILE_DEFINITIONS
    SDCPP_BUILD_COMMIT=${SDCPP_BUILD_COMMIT}
    SDCPP_BUILD_VERSION=${SDCPP_BUILD_VERSION}
)

# ─── stable-diffusion 库目标 ─────────────────────────────────
set(SD_LIB stable-diffusion)

if(SD_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF)
    add_library(${SD_LIB} SHARED ${SD_LIB_SOURCES})
    add_definitions(-DSD_BUILD_SHARED_LIB)
    target_compile_definitions(${SD_LIB} PRIVATE -DSD_BUILD_DLL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    set(BUILD_SHARED_LIBS OFF)
    add_library(${SD_LIB} STATIC ${SD_LIB_SOURCES})
endif()

if(SD_SYCL)
    if(WIN32)
        target_compile_options(${SD_LIB} PRIVATE /fp:precise)
    else()
        target_compile_options(${SD_LIB} PRIVATE -fp-model=precise)
    endif()
endif()

# ─── 链接依赖 ────────────────────────────────────────────────
# SD 链接统一的 ggml（由根 CMakeLists 编译，USE_ACL_GRAPH 已启用）
# thirdparty (zip 等) 由 sd/src 自己管理
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty thirdparty-sd-build)

target_link_libraries(${SD_LIB} PUBLIC ggml zip)
target_include_directories(${SD_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)
target_compile_features(${SD_LIB} PUBLIC c_std_11 cxx_std_17)

# ─── 安装 ────────────────────────────────────────────────────
set(SD_PUBLIC_HEADERS ${CMAKE_SOURCE_DIR}/include/stable-diffusion.h)
set_target_properties(${SD_LIB} PROPERTIES PUBLIC_HEADER "${SD_PUBLIC_HEADERS}")
install(TARGETS ${SD_LIB} LIBRARY PUBLIC_HEADER)
