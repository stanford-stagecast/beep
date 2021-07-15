cc_binary(
    name = "beep-example",
    srcs = ["beep-example.cc"],
    deps = [
        "//beep/audio",
        "//beep/models:test_model",
        "//beep/stats",
        "//beep/input",
        "//beep/util",
        "@alsa//:lib",
        "@libtorch_archive//:torch",
        "@org_tensorflow//third_party/eigen3",
        "@sndfile//:lib",
    ],
)
