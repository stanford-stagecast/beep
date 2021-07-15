workspace(name = "beep")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_pkg_config",
    strip_prefix = "bazel_pkg_config-master",
    urls = ["https://github.com/cherrry/bazel_pkg_config/archive/master.zip"],
)

load("@bazel_pkg_config//:pkg_config.bzl", "pkg_config")

pkg_config(
    name = "alsa",
)

pkg_config(
    name = "sndfile",
)

local_repository(
    name = "org_tensorflow",
    path = "third_party/tensorflow",
)

load(
    "@org_tensorflow//tensorflow:version_check.bzl",
    "check_bazel_version_at_least",
)

check_bazel_version_at_least("3.7.2")

load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")

workspace()

http_archive(
    name = "libtorch_archive",
    strip_prefix = "libtorch",
    type = "zip",
    urls = ["https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip"],
    sha256 = "6b99edc046f37ad37a3b04dc86663895f10c362af09fdd10884f30360d0ba023",
    build_file = "@//:libtorch.BUILD"
)
