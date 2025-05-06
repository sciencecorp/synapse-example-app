cmake_minimum_required(VERSION 3.20.1)
project(synapse-example-app)

include("cmake/protos.cmake")

option(BUILD_FOR_ARM64 "Build for ARM64 architecture" OFF)
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

find_library(SYNAPSE_APP_SDK_LIB NAMES synapse-app-sdk libsynapse-app-sdk.so.0.1.0)

# Gather all the sources
file(GLOB_RECURSE EXAMPLE_APP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(SYNAPSE_EXAMPLE_APP_TARGET synapse-example-app)
# Add the sources to the build
add_executable(${SYNAPSE_EXAMPLE_APP_TARGET}
    ${EXAMPLE_APP_SOURCES}
)

generate_protobufs(
    TARGET ${SYNAPSE_EXAMPLE_APP_TARGET}
    OUT_PROTO_DIR PROTO_OUT_DIR
)

# Be strict about the standard
set_target_properties(${SYNAPSE_EXAMPLE_APP_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# Add the include directory
target_include_directories(${SYNAPSE_EXAMPLE_APP_TARGET}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link dependencies
target_link_libraries(${SYNAPSE_EXAMPLE_APP_TARGET}
    PUBLIC
        cppzmq
        protobuf::libprotobuf
        iir::iir
        ${SYNAPSE_APP_SDK_LIB}

    PRIVATE
        spdlog::spdlog
)
