@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk

set TARGET_CONFIG_FILE=config_vicuna_7b_spec_dec_target.yaml
set DRAFT_CONFIG_FILE=config_vicuna_160m_spec_dec_draft.yaml
set INPUT_PROMPT=mt_bench_1st_turn.txt
set MAX_RESPONSE=512
set PREFORMATTER=VicunaNoInput
set TARGET_TEMPERATURE=0.0
set DRAFT_TEMPERATURE=0.0

:: Push yaml config file
adb push %INPUT_PROMPT% %PHONE_PATH%
adb push %TARGET_CONFIG_FILE% %PHONE_PATH%
adb push %DRAFT_CONFIG_FILE% %PHONE_PATH%

:: Set execute permission
adb shell "chmod +x %PHONE_PATH%/main_spec_dec"

:: Run using the below commands
set INFER_TYPE=0
set DRAFT_LENGTH=5
adb shell "cd %PHONE_PATH%; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD ./main_spec_dec %TARGET_CONFIG_FILE% --infer-type %INFER_TYPE% --draft %DRAFT_CONFIG_FILE% --draft-len %DRAFT_LENGTH% -i %INPUT_PROMPT% -m %MAX_RESPONSE% --preformatter %PREFORMATTER% --one-prompt-per-line --target-temperature %TARGET_TEMPERATURE% --draft-temperature %DRAFT_TEMPERATURE%"

:error