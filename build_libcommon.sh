#!/bin/bash
set -e

LIB_LLM_SRC_PATH=jni/prebuilt/src

ndk-build -j 8 NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=${LIB_LLM_SRC_PATH}/common/Android.mk NDK_APPLICATION_MK=${LIB_LLM_SRC_PATH}/Application.mk

cp -f libs/arm64-v8a/libcommon.so jni/prebuilt/

mkdir -p jni/prebuilt/include/common
cp -f ${LIB_LLM_SRC_PATH}/common/*.h jni/prebuilt/include/common/
