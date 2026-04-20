@echo off
setlocal

IF "%1"=="-usdk" (
    SET EXTRA_FLAGS=-DUSE_USDK_BACKEND
    SHIFT
)

@echo on

set LIB_LLM_SRC_PATH=jni\prebuilt\src

:: Build shared library
call ndk-build -j 8 NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=%LIB_LLM_SRC_PATH%\Android_mllm.mk NDK_APPLICATION_MK=%LIB_LLM_SRC_PATH%\Application_mllm.mk EXTRA_GLOBAL_CPPFLAGS=%EXTRA_FLAGS% || goto :error

:: Copy .so to prebuilt directory
xcopy /s/y libs\arm64-v8a\libmtk_llm.so jni\prebuilt\
xcopy /s/y libs\arm64-v8a\libmtk_mllm.so jni\prebuilt\
xcopy /s/y libs\arm64-v8a\libopencv_java4.so jni\prebuilt\

:: Copy header files to prebuilt include directory
xcopy /s/y %LIB_LLM_SRC_PATH%\mtk_llm_types.h jni\prebuilt\include
xcopy /s/y %LIB_LLM_SRC_PATH%\mtk_llm_options.h jni\prebuilt\include
xcopy /s/y %LIB_LLM_SRC_PATH%\mtk_llm.h jni\prebuilt\include
xcopy /s/y %LIB_LLM_SRC_PATH%\mtk_mllm.h jni\prebuilt\include
xcopy /s/y %LIB_LLM_SRC_PATH%\medusa_config.h jni\prebuilt\include

:error