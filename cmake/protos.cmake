function(generate_protobufs)
  cmake_parse_arguments(PARSE_ARGV 0 "arg"
      ""
      "TARGET;OUT_PROTO_DIR"
      "PROTO_DIRS;PROTO_FILES"
  )

  if(DEFINED arg_UNPARSED_ARGUMENTS)
    message(WARNING "generate_protobufs was passed extra arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT DEFINED arg_OUT_PROTO_DIR)
    message(FATAL_ERROR "OUT_PROTO_DIR must be specified.")
  endif()
  if(NOT DEFINED arg_TARGET)
    message(FATAL_ERROR "TARGET must be specified.")
  endif()
  if(NOT DEFINED arg_PROTO_DIRS AND NOT DEFINED arg_PROTO_FILES)
    message(FATAL_ERROR "At least one of PROTO_DIRS or PROTO_FILES must be specified.")
  endif()

  set(PROTO_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
  file(MAKE_DIRECTORY ${PROTO_OUT_DIR})

  # Initialize empty lists for proto include dirs and proto files
  set(PROTO_INCLUDE_DIRS "")
  set(PROTOS "")

  # Include Synapse API proto directory in the include path
  get_filename_component(SYNAPSE_PROTO_INCLUDE_DIR ./external/sciencecorp/synapse-api REALPATH)
  list(APPEND PROTO_INCLUDE_DIRS ${SYNAPSE_PROTO_INCLUDE_DIR})
  if(DEFINED SCIFI_PROTO_INCLUDE_DIR)
    list(APPEND PROTO_INCLUDE_DIRS ${SCIFI_PROTO_INCLUDE_DIR})
  endif()

  # Add custom proto directories and files if provided
  if(DEFINED arg_PROTO_DIRS)
    foreach(DIR ${arg_PROTO_DIRS})
      get_filename_component(ABS_DIR ${DIR} REALPATH)
      list(APPEND PROTO_INCLUDE_DIRS ${ABS_DIR})
      # If specific files not provided, include all protos in the directory
      if(NOT DEFINED arg_PROTO_FILES)
        file(GLOB_RECURSE DIR_PROTOS ${ABS_DIR}/*.proto)
        list(APPEND PROTOS ${DIR_PROTOS})
      endif()
    endforeach()
  endif()

  # Add specific proto files if provided
  if(DEFINED arg_PROTO_FILES)
    foreach(PROTO_FILE ${arg_PROTO_FILES})
      get_filename_component(ABS_PROTO_FILE ${PROTO_FILE} REALPATH)
      list(APPEND PROTOS ${ABS_PROTO_FILE})
    endforeach()
  endif()

  protobuf_generate(
    TARGET ${arg_TARGET}
    LANGUAGE cpp
    IMPORT_DIRS ${PROTO_INCLUDE_DIRS}
    PROTOS ${PROTOS}
    PROTOC_OUT_DIR ${PROTO_OUT_DIR}
    OUT_VAR PROTO_SOURCES
  )

# NOTE: Uncomment this to generate the gRPC code if we ever need it
#   protobuf_generate(
#     TARGET ${arg_TARGET}
#     LANGUAGE grpc
#     IMPORT_DIRS ${SYNAPSE_PROTO_INCLUDE_DIR}
#     PROTOS ${SYNAPSE_PROTO_INCLUDE_DIR}/api/synapse.proto
#     GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
#     PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
#     PROTOC_OUT_DIR ${PROTO_OUT_DIR}
#     OUT_VAR PROTO_SOURCES
#   )

  set("${arg_OUT_PROTO_DIR}" "${PROTO_OUT_DIR}" PARENT_SCOPE)

endfunction()
