# Beep

## Build instructions

Following packages are required to build `beep`:

* `bazel` — [install instructions for Ubuntu](https://docs.bazel.build/versions/main/install-ubuntu.html)
* `gcc-9` — ⚠️ doesn't work with GCC 10
* `libalsa-dev`
* `libsndfile-dev`

0. Install the dependencies.

1. Clone this repo:

```
git clone git@github.com:stanford-stagecast/beep
```

2. Run `./prepare-submodules.sh` script. This will fetch and patch TensorFlow
submodule.

3. Run the following command to build the project (first-time build could take ~2 hours
on an 8-core machine).

```
bazel build //beep/frontend:beep-example
```

4. Start the program by running:

```
./bazel-bin/beep/frontend/beep-example <PATH-TO-WAV-FILE>
```

The supplied WAV file should be played out of the LEFT channel of your headphones!
