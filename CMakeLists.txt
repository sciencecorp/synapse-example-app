cmake_minimum_required(VERSION 3.20.1)
project(synapse-example-app)

include("cmake/protos.cmake")

option(BUILD_FOR_ARM64 "Build for ARM64 architecture" OFF)
option(USE_LOCAL_SDK "Use locally built SDK instead of system installation" OFF)
set(LOCAL_SDK_PATH "app-sdk" CACHE PATH "Path to local SDK build directory")

if (BUILD_FOR_ARM64 AND NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  message(STATUS "Cross compiling for ARM64")
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
endif()

# Required packages
find_package(cppzmq REQUIRED)
find_package(Protobuf REQUIRED)
find_package(spdlog REQUIRED CONFIG)
find_package(iir REQUIRED CONFIG)

if(true)
    if(NOT LOCAL_SDK_PATH)
        message(FATAL_ERROR "USE_LOCAL_SDK is ON but LOCAL_SDK_PATH is not set")
    endif()
    message(STATUS "Using local SDK from: ${LOCAL_SDK_PATH}")

    set(SDK_LIB_PATH "${LOCAL_SDK_PATH}")
    set(SDK_LIB_NAME "synapse-app-sdk")
    link_directories(${SDK_LIB_PATH})

    set(SYNAPSE_APP_SDK_LIB ${SDK_LIB_NAME})

    include_directories("${LOCAL_SDK_PATH}/include")
else()
    find_library(SYNAPSE_APP_SDK_LIB NAMES synapse-app-sdk libsynapse-app-sdk.so.0.1.0)
endif()

message(STATUS "SYNAPSE_APP_SDK_LIB: ${SYNAPSE_APP_SDK_LIB}")

add_executable(synapse-example-app
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fixed_weight_decoder.cpp
)

generate_protobufs(
    TARGET synapse-example-app
    OUT_PROTO_DIR PROTO_OUT_DIR
    PROTO_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/external/sciencecorp/synapse-api"
)

# Be strict about the standard
set_target_properties(synapse-example-app PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(synapse-example-app
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
   PRIVATE
        ${PROTO_OUT_DIR}
)

# Link dependencies
target_link_libraries(synapse-example-app
    PUBLIC
        cppzmq
        protobuf::libprotobuf
        iir::iir
        ${SYNAPSE_APP_SDK_LIB}

    PRIVATE
        spdlog::spdlog
)

# Packaging configuration
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/dist")
include(CPack)
