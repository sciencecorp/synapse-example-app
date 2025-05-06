function(generate_protobufs)
  cmake_parse_arguments(PARSE_ARGV 0 "arg"
      ""
      "TARGET;OUT_PROTO_DIR"
      ""
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

  set(PROTO_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
  file(MAKE_DIRECTORY ${PROTO_OUT_DIR})

  get_filename_component(SYNAPSE_PROTO_INCLUDE_DIR ./external/sciencecorp/synapse-api REALPATH)
  file(GLOB_RECURSE SYNAPSE_PROTOS ${SYNAPSE_PROTO_INCLUDE_DIR}/*.proto)

  set(PROTO_INCLUDE_DIRS ${SYNAPSE_PROTO_INCLUDE_DIR} ${SCIFI_PROTO_INCLUDE_DIR})
  set(PROTOS ${SYNAPSE_PROTOS} ${SCIFI_PROTOS})

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
