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
    urls = ["https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.9.0%2Bcpu.zip"],
    sha256 = "65192e6e1c3046265dc100003d494c415342c284800cc58db4933e370d98a0db",
    build_file = "@//:libtorch.BUILD"
)
