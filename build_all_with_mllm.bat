@echo on

call build_clean.bat

call build_libcommon.bat        || goto :error
call build_libtokenizer.bat     || goto :error

call build_libmllm.bat         || goto :error
call build_runner.bat          || goto :error
call build_mllm_runner.bat     || goto :error

:error