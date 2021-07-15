load("@org_tensorflow//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

tf_library(
    name = "test_model",
    config = "config.pbtxt",
    cpp_class = "TestModel",
    graph = "frozen_graph.pb",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "models",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.hh"]),
    visibility = ["//visibility:public"],
    deps = [
        ":test_model",
        "//beep/util:util",
        "@libtorch_archive//:torch"
    ],
)
