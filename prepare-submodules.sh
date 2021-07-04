#!/bin/bash -ex

git submodule update --recursive --init third_party/tensorflow
git submodule foreach git reset --hard

git apply tfcompile.patch
