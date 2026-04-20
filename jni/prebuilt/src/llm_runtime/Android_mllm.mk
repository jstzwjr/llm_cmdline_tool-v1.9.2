LOCAL_PATH := $(call my-dir)
USER_LOCAL_C_INCLUDES := $(LOCAL_C_INCLUDES)

include $(CLEAR_VARS)
LOCAL_MODULE := mllm_runtime
LOCAL_SRC_FILES := mllm_runtime.cpp llava_mllm_runtime.cpp
LOCAL_STATIC_LIBRARIES += backend executor llm_helper image_utils llm_runtime
LOCAL_SHARED_LIBRARIES += common
LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES)
include $(BUILD_STATIC_LIBRARY)

LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES)