cmake_minimum_required(VERSION 3.20.1)
project(synapse-example-app)

option(BUILD_FOR_ARM64 "Build for ARM64 architecture" OFF)
if (BUILD_FOR_ARM64 AND NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  message(STATUS "Cross compiling for ARM64")
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
  set(Protobuf_PROTOC_EXECUTABLE "/usr/local/bin/protoc")
  message(STATUS "Explicitly using protoc at: ${Protobuf_PROTOC_EXECUTABLE}")
endif()

list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/cmake")
list(APPEND CMAKE_PREFIX_PATH "/usr/local/share/cmake")
list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/synapse-app-sdk")
list(APPEND CMAKE_MODULE_PATH "/usr/local/share/cmake/synapse-app-sdk")

# Required packages
find_package(spdlog REQUIRED CONFIG)
find_package(synapse-app-sdk REQUIRED)

# Gather all the sources
file(GLOB_RECURSE EXAMPLE_APP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(SYNAPSE_EXAMPLE_APP_TARGET synapse-example-app)

# Add the sources to the build
add_executable(${SYNAPSE_EXAMPLE_APP_TARGET}
    ${EXAMPLE_APP_SOURCES}
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
        synapse::synapse-app-sdk
    PRIVATE
        spdlog::spdlog
)
