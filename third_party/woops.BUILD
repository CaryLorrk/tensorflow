licenses([
    "notice",
])

load("@org_tensorflow//:tensorflow/core/platform/default/build_config.bzl", "cc_proto_library")

cc_proto_library(
    name = "woops_grpc",
    srcs = glob(["**/*.proto"]),
    protoc = "@protobuf_archive//:protoc",
    use_grpc_plugin = True,
)

cc_library(
    name = "woops",
    srcs = glob(["**/*.cc"]),
    hdrs = glob(["**/*.h"]),
    includes = [ "." ],
    copts = ["-std=c++14"],
    deps = ["@grpc//:grpc++_unsecure",
            ":woops_grpc"],
    visibility = ["//visibility:public"],
)
