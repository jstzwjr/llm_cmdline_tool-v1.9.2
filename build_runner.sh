#!/bin/bash
set -e

ndk-build -j 8

PHONE_PATH=/data/local/tmp/llm_sdk

adb shell mkdir -p $PHONE_PATH

adb push libs/arm64-v8a/main $PHONE_PATH/
adb push libs/arm64-v8a/main_batch_gen $PHONE_PATH/
adb push libs/arm64-v8a/main_spec_dec $PHONE_PATH/
adb push libs/arm64-v8a/main_medusa $PHONE_PATH/
adb push libs/arm64-v8a/main_tree_spec_dec_plus $PHONE_PATH/
adb push libs/arm64-v8a/libmtk_llm.so $PHONE_PATH/
adb push jni/prebuilt/libcommon.so $PHONE_PATH/
adb push jni/prebuilt/libtokenizer.so $PHONE_PATH/
adb push libs/arm64-v8a/libc++_shared.so $PHONE_PATH/
adb push libs/arm64-v8a/libyaml-cpp.so $PHONE_PATH/

adb shell "chmod +x $PHONE_PATH/main"
adb shell "chmod +x $PHONE_PATH/main_spec_dec"
