@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk

set CONFIG_FILE=config_medusa_vicuna_7b_3heads_8t.yaml
set INPUT_PROMPT=mt_bench_1st_turn.txt
set MAX_RESPONSE=512
set PREFORMATTER=VicunaNoInput
set TEMPERATURE=0.7

:: Push yaml config file
adb push %CONFIG_FILE% %PHONE_PATH%
adb push %INPUT_PROMPT% %PHONE_PATH%

:: Set execute permission
adb shell "chmod +x %PHONE_PATH%/main_medusa"

:: Run using the below commands
adb shell "cd %PHONE_PATH%; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD ./main_medusa %CONFIG_FILE% -i %INPUT_PROMPT% -m %MAX_RESPONSE% --preformatter %PREFORMATTER% --one-prompt-per-line --temperature %TEMPERATURE%"

:error