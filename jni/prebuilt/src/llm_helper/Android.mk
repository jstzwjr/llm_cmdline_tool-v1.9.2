LOCAL_PATH := $(call my-dir)
USER_LOCAL_C_INCLUDES := $(LOCAL_C_INCLUDES)

include $(CLEAR_VARS)
LOCAL_MODULE := mtk_cache_eviction
LOCAL_SRC_FILES := ../../libmtk_cache_eviction.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := llm_helper
LOCAL_SRC_FILES := ringbuffer_cache_manager.cpp \
                   rotary_embedding.cpp         \
                   mask_builder.cpp             \
				   token_embedding.cpp			\
                   lora_weights_loader.cpp
LOCAL_SHARED_LIBRARIES += common
LOCAL_STATIC_LIBRARIES += mtk_cache_eviction
LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES)
include $(BUILD_STATIC_LIBRARY)

LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES)