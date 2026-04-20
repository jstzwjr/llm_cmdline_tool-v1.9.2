@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk

set CONFIG_FILE=config_qwen2.5_3b_cache_eviction.yaml
set INPUT_PROMPT=longbench_vcsum.txt
set MAX_RESPONSE=512
set PREFORMATTER=QwenNoInput

:: Push yaml config file
adb push %CONFIG_FILE% %PHONE_PATH%
adb push %INPUT_PROMPT% %PHONE_PATH%

:: Set execute permission
adb shell "chmod +x %PHONE_PATH%/main"

:: Run using the below commands
adb shell "cd %PHONE_PATH%; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD ./main %CONFIG_FILE% -i %INPUT_PROMPT% --preformatter %PREFORMATTER% -m %MAX_RESPONSE% --one-prompt-per-line"

:error