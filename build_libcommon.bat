@echo on

set LIB_LLM_SRC_PATH=jni\prebuilt\src

:: Build shared library for llm_helper
call ndk-build -j 8 NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=%LIB_LLM_SRC_PATH%\common\Android.mk NDK_APPLICATION_MK=%LIB_LLM_SRC_PATH%\Application.mk || goto :error

:: Copy .so to prebuilt directory
xcopy /s/y libs\arm64-v8a\libcommon.so jni\prebuilt\

:: Copy header files
mkdir jni\prebuilt\include\common 2> NUL
xcopy /s/y %LIB_LLM_SRC_PATH%\common\*.h jni\prebuilt\include\common

:error