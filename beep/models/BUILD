load("@org_tensorflow//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

tf_library(
    name = "test_model",
    config = "config.pbtxt",
    cpp_class = "TestModel",
    graph = "frozen_graph.pb",
    visibility = ["//visibility:public"],
)
