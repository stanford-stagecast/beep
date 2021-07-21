cc_library(
  name = "torch",
  srcs = [
    "lib/libtorch.so",
    "lib/libtorch_cpu.so",
    "lib/libc10.so",
    "lib/libgomp-a34b3233.so.1",
  ],
  linkopts = [
    "-ltorch",
    "-ltorch_cpu",
    "-lc10",
  ],
  hdrs = glob(["include/**/*.h"]),
  includes = [
    "include",
    "include/torch/csrc/api/include",
  ],
  copts = ["-D_GLIBCXX_USE_CXX11_ABI=0"],
  visibility = ["//visibility:public"]
)
