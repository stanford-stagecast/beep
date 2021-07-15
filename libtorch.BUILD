cc_library(
  name = "torch",
  srcs = [
    "lib/libtorch.so",
    "lib/libtorch_cpu.so",
    "lib/libc10.so",
    "lib/libgomp-75eea7e8.so.1"
  ],
  hdrs = glob(["include/**/*.h"]),
  includes = [
    "include",
    "include/torch/csrc/api/include",
  ],
  visibility = ["//visibility:public"]
)
