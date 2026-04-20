@echo on

call build_clean.bat

call build_libcommon.bat        || goto :error
call build_libtokenizer.bat     || goto :error
call build_libllm.bat -usdk     || goto :error
call build_runner.bat           || goto :error

:error