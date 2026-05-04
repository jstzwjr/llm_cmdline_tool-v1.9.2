#!/bin/bash
set -e

LIB_LLM_SRC_PATH=jni/prebuilt/src

ndk-build -j 8 NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=${LIB_LLM_SRC_PATH}/tokenizer/Android_SharedLib.mk NDK_APPLICATION_MK=${LIB_LLM_SRC_PATH}/Application.mk

cp -f libs/arm64-v8a/libtokenizer.so jni/prebuilt/

TOKENIZER_INCLUDE_PATH=jni/prebuilt/include/tokenizer
mkdir -p ${TOKENIZER_INCLUDE_PATH}
cp -f ${LIB_LLM_SRC_PATH}/tokenizer/tokenizer.h ${TOKENIZER_INCLUDE_PATH}/
cp -f ${LIB_LLM_SRC_PATH}/tokenizer/tokenizer_factory.h ${TOKENIZER_INCLUDE_PATH}/
cp -f ${LIB_LLM_SRC_PATH}/tokenizer/sentencepiece_tokenizer.h ${TOKENIZER_INCLUDE_PATH}/
cp -f ${LIB_LLM_SRC_PATH}/tokenizer/huggingface_tokenizer.h ${TOKENIZER_INCLUDE_PATH}/
cp -f ${LIB_LLM_SRC_PATH}/tokenizer/tiktoken_tokenizer.h ${TOKENIZER_INCLUDE_PATH}/

PHONE_PATH=/data/local/tmp/llm_sdk
adb shell mkdir -p $PHONE_PATH
adb push libs/arm64-v8a/libsentencepiece.so $PHONE_PATH/
adb push libs/arm64-v8a/libhf-tokenizer.so $PHONE_PATH/
adb push libs/arm64-v8a/libre2.so $PHONE_PATH/
