cc_binary(
    name = "beep-example",
    srcs = ["beep-example.cc"],
    deps = [
        "//beep/audio",
        "//beep/models:test_model",
        "//beep/stats",
        "//beep/util",
        "@alsa//:lib",
        "@org_tensorflow//third_party/eigen3",
        "@sndfile//:lib",
    ],
)
